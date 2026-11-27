import json
import platform
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class KvasirDataset(Dataset):

    def __init__(self, data_dir, stem_list, augment=False, img_size=256):
        self.data_dir   = Path(data_dir)
        self.stems      = stem_list
        self.img_size   = img_size
        self.do_augment = augment

        with open(self.data_dir / "kavsir_bboxes.json") as f:
            self.bboxes = json.load(f)

        # Import once — fail clearly if missing
        from preprocessing import preprocess_image
        from feature_extraction import build_tensor
        self._preprocess  = preprocess_image
        self._build_tensor = build_tensor

        self._aug_fn = None
        if augment:
            try:
                from augmentation import augment as _aug
                self._aug_fn = _aug
            except ImportError:
                print("[WARNING] augmentation.py not found — running without augmentation.")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem      = self.stems[idx]
        img_path  = str(self.data_dir / "images" / f"{stem}.jpg")
        mask_ext = ".jpg" if (self.data_dir / "masks" / f"{stem}.jpg").exists() else ".png"
        mask_path = str(self.data_dir / "masks" / f"{stem}{mask_ext}")

        #  Image: preprocess -> feature extraction -> (H,W,9) 
        bgr_float  = self._preprocess(img_path)            # (H,W,3) float32
        bbox_entry = self.bboxes.get(stem, {"height": None, "width": None, "bbox": []})
        tensor_hwc = self._build_tensor(bgr_float, bbox_entry)  # (H,W,9) float32

        #  Mask: load, resize, binarise -> (H,W) 
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)
        mask_hw = (mask > 127).astype(np.float32)           # (H,W)

        #  Augmentation (train only) 
        if self.do_augment and self._aug_fn is not None:
            tensor_hwc, mask_hw = self._aug_fn(tensor_hwc, mask_hw)

        #  Convert to (C,H,W) for PyTorch 
        tensor_chw = tensor_hwc.transpose(2, 0, 1)          # (9,H,W)
        mask_1hw   = mask_hw[np.newaxis]                     # (1,H,W)

        return (
            torch.from_numpy(tensor_chw.copy()),
            torch.from_numpy(mask_1hw.copy()),
        )



def get_dataloaders(
    data_dir,
    split_file  = "split_index.json",
    batch_size  = 16,
    num_workers = 4,
    img_size    = 256,
    pin_memory  = None,
):
    with open(split_file) as f:
        raw = json.load(f)

    split = {
        "train": raw.get("train", raw.get("train_ids", [])),
        "val":   raw.get("val",   raw.get("val_ids",   [])),
        "test":  raw.get("test",  raw.get("test_ids",  [])),
    }

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if platform.system() == "Windows":
        num_workers = 0

    def _loader(stems, augment, shuffle):
        ds = KvasirDataset(data_dir, stems, augment=augment, img_size=img_size)
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = augment,
        )

    train_loader = _loader(split["train"], augment=True,  shuffle=True)
    val_loader   = _loader(split["val"],   augment=False, shuffle=False)
    test_loader  = _loader(split["test"],  augment=False, shuffle=False)

    print(f"DataLoaders ready — "
          f"train={len(split['train'])}  "
          f"val={len(split['val'])}  "
          f"test={len(split['test'])}")

    return train_loader, val_loader, test_loader