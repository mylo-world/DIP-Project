"""
ablation.py — Ablation study for Kvasir-SEG polyp segmentation.

Trains four variants to isolate the contribution of each component:

  V1: Baseline   — RGB only (3ch), no CLAHE
  V2: +CLAHE     — RGB + CLAHE (3ch), no extra features
  V3: +Features  — RGB + CLAHE + LBP + Canny + HSV (8ch), no BBox
  V4: Full model — RGB + CLAHE + LBP + Canny + HSV + BBox (9ch)

V4 is already trained; its checkpoint is loaded directly if it exists.

Usage:
    python ablation.py --data_dir Kvasir-SEG --epochs 30
    python ablation.py --data_dir Kvasir-SEG --epochs 30 --skip_v4

Results saved to:
    ablation_results/
    ├── v1_baseline/         checkpoint + history
    ├── v2_clahe/
    ├── v3_features/
    ├── v4_full/
    ├── ablation_summary.json
    └── ablation_table.txt   (print-ready comparison table)
"""

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

# ── Import team modules ───────────────────────────────────────────────────────
from preprocessing import preprocess_image          # A1
from feature_extraction import (                    # A1
    extract_lbp, extract_canny, extract_hsv, build_bbox_map
)
from model import UNetMini                          # A2
from losses import CombinedLoss, dice_coefficient   # A3
from evaluate import evaluate                       # A3


# ═════════════════════════════════════════════════════════════════════════════
# Per-variant feature builders
# Each returns a (H,W,C) float32 numpy array — matches A2's augment() interface
# ═════════════════════════════════════════════════════════════════════════════

def _load_mask(mask_path, size=256):
    """Load and binarise mask; auto-detect .jpg/.png extension."""
    p = Path(mask_path)
    if not p.exists():
        # Try alternate extension
        alt = p.with_suffix(".png" if p.suffix == ".jpg" else ".jpg")
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.float32)


def build_v1(img_path, bbox_entry, size=256):
    """V1: Raw RGB only — no CLAHE, no extra features. (3ch)"""
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = img[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB, normalise
    return rgb                                           # (H,W,3)


def build_v2(img_path, bbox_entry, size=256):
    """V2: RGB + CLAHE only. (3ch)"""
    bgr_float = preprocess_image(str(img_path))         # CLAHE + normalise
    rgb = bgr_float[:, :, ::-1]                         # BGR→RGB
    return rgb.astype(np.float32)                       # (H,W,3)


def build_v3(img_path, bbox_entry, size=256):
    """V3: RGB + CLAHE + LBP + Canny + HSV, no BBox. (8ch)"""
    bgr_float = preprocess_image(str(img_path))
    rgb   = bgr_float[:, :, ::-1]                       # (H,W,3)
    lbp   = extract_lbp(bgr_float)[:, :, np.newaxis]    # (H,W,1)
    canny = extract_canny(bgr_float)[:, :, np.newaxis]  # (H,W,1)
    h, s, v = extract_hsv(bgr_float)
    hsv   = np.stack([h, s, v], axis=2)                 # (H,W,3)
    return np.concatenate([rgb, lbp, canny, hsv], axis=2).astype(np.float32)  # (H,W,8)


def build_v4(img_path, bbox_entry, size=256):
    """V4: Full 9-channel model (RGB+CLAHE+LBP+Canny+HSV+BBox)."""
    bgr_float = preprocess_image(str(img_path))
    rgb   = bgr_float[:, :, ::-1]
    lbp   = extract_lbp(bgr_float)[:, :, np.newaxis]
    canny = extract_canny(bgr_float)[:, :, np.newaxis]
    h, s, v = extract_hsv(bgr_float)
    hsv   = np.stack([h, s, v], axis=2)
    bbox  = build_bbox_map(bbox_entry, target_size=size)[:, :, np.newaxis]
    return np.concatenate([rgb, lbp, canny, hsv, bbox], axis=2).astype(np.float32)  # (H,W,9)


VARIANTS = {
    "v1_baseline":  {"builder": build_v1, "in_channels": 3,
                     "label": "V1: Baseline (RGB only)"},
    "v2_clahe":     {"builder": build_v2, "in_channels": 3,
                     "label": "V2: +CLAHE"},
    "v3_features":  {"builder": build_v3, "in_channels": 8,
                     "label": "V3: +CLAHE+LBP+Canny+HSV"},
    "v4_full":      {"builder": build_v4, "in_channels": 9,
                     "label": "V4: Full (proposed)"},
}


# ═════════════════════════════════════════════════════════════════════════════
# Dataset — variant-aware
# ═════════════════════════════════════════════════════════════════════════════

class AblationDataset(Dataset):
    def __init__(self, data_dir, stem_list, builder_fn,
                 augment=False, img_size=256):
        self.data_dir  = Path(data_dir)
        self.stems     = stem_list
        self.builder   = builder_fn
        self.img_size  = img_size
        self.do_augment = augment

        with open(self.data_dir / "kavsir_bboxes.json") as f:
            self.bboxes = json.load(f)

        self._aug_fn = None
        if augment:
            try:
                from augmentation import augment as _aug
                self._aug_fn = _aug
            except ImportError:
                pass

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem      = self.stems[idx]
        img_path  = self.data_dir / "images" / f"{stem}.jpg"
        mask_ext  = ".jpg" if (self.data_dir / "masks" / f"{stem}.jpg").exists() else ".png"
        mask_path = self.data_dir / "masks" / f"{stem}{mask_ext}"

        bbox_entry = self.bboxes.get(stem, {"height": None, "width": None, "bbox": []})

        tensor_hwc = self.builder(img_path, bbox_entry, self.img_size)  # (H,W,C)
        mask_hw    = _load_mask(mask_path, self.img_size)               # (H,W)

        if self.do_augment and self._aug_fn is not None:
            tensor_hwc, mask_hw = self._aug_fn(tensor_hwc, mask_hw)

        tensor_chw = tensor_hwc.transpose(2, 0, 1)      # (C,H,W)
        mask_1hw   = mask_hw[np.newaxis]                 # (1,H,W)

        return (torch.from_numpy(tensor_chw.copy()),
                torch.from_numpy(mask_1hw.copy()))


def _make_loaders(data_dir, split_file, builder_fn,
                  batch_size, img_size):
    with open(split_file) as f:
        raw = json.load(f)
    split = {
        "train": raw.get("train", raw.get("train_ids", [])),
        "val":   raw.get("val",   raw.get("val_ids",   [])),
        "test":  raw.get("test",  raw.get("test_ids",  [])),
    }
    nw = 0 if platform.system() == "Windows" else 4
    pin = torch.cuda.is_available()

    def _loader(stems, aug, shuffle):
        ds = AblationDataset(data_dir, stems, builder_fn,
                             augment=aug, img_size=img_size)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=nw, pin_memory=pin,
                          drop_last=aug)

    return (
        _loader(split["train"], aug=True,  shuffle=True),
        _loader(split["val"],   aug=False, shuffle=False),
        _loader(split["test"],  aug=False, shuffle=False),
        split,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Training loop (self-contained, no dependency on train.py)
# ═════════════════════════════════════════════════════════════════════════════

class _EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = -float("inf")

    def step(self, val_dice):
        if val_dice > self.best + self.min_delta:
            self.best    = val_dice
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def _run_epoch(model, loader, criterion, device, optimizer=None):
    model.train() if optimizer else model.eval()
    total_loss = total_dice = 0.0
    ctx = torch.enable_grad() if optimizer else torch.no_grad()
    with ctx:
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
            logits = model(images)
            loss   = criterion(logits, masks)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            total_dice += dice_coefficient(logits, masks)
    n = len(loader)
    return total_loss / n, total_dice / n


def train_variant(variant_name, variant_cfg, args, device, split_file):
    """Train one ablation variant. Returns history dict."""
    label      = variant_cfg["label"]
    in_ch      = variant_cfg["in_channels"]
    builder_fn = variant_cfg["builder"]
    ckpt_dir   = Path(args.output_dir) / variant_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {label}  |  in_channels={in_ch}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader, _ = _make_loaders(
        args.data_dir, split_file, builder_fn,
        args.batch_size, img_size=256
    )

    model     = UNetMini(in_channels=in_ch, out_channels=1).to(device)
    criterion = CombinedLoss(smooth=1.0, bce_weight=0.25, pos_weight=6.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                  factor=0.5, patience=7)
    stopper   = _EarlyStopping(patience=args.patience)

    history = {"train_loss": [], "val_loss": [],
               "train_dice": [], "val_dice": [],
               "best_epoch": 0, "best_val_dice": 0.0}

    best_dice = -float("inf")
    print(f"{'Ep':>4} {'TrLoss':>8} {'VaLoss':>8} "
          f"{'TrDice':>8} {'VaDice':>8}  {'t':>5}")
    print("-" * 48)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_dice = _run_epoch(model, train_loader, criterion,
                                      device, optimizer)
        va_loss, va_dice = _run_epoch(model, val_loader,   criterion,
                                      device)
        scheduler.step(va_dice)

        history["train_loss"].append(round(tr_loss, 6))
        history["val_loss"].append(round(va_loss, 6))
        history["train_dice"].append(round(tr_dice, 6))
        history["val_dice"].append(round(va_dice, 6))

        print(f"{epoch:>4} {tr_loss:>8.4f} {va_loss:>8.4f} "
              f"{tr_dice:>8.4f} {va_dice:>8.4f}  "
              f"{time.time()-t0:>4.1f}s")

        if va_dice > best_dice:
            best_dice = va_dice
            history["best_epoch"]    = epoch
            history["best_val_dice"] = round(va_dice, 4)
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_dice": va_dice},
                       ckpt_dir / "best_model.pth")
            print(f"       ✓ saved (val_dice={va_dice:.4f})")

        if stopper.step(va_dice):
            print(f"  Early stop at epoch {epoch}.")
            break

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Evaluate on test set ──────────────────────────────────────────
    print(f"\n  Evaluating {variant_name} on test set...")
    ckpt = torch.load(ckpt_dir / "best_model.pth", map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_summary = evaluate(
        model, test_loader,
        device      = str(device),
        threshold   = 0.5,
        results_dir = str(ckpt_dir / "test_results"),
    )
    history["test_metrics"] = test_summary
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


# ═════════════════════════════════════════════════════════════════════════════
# V4 special case — load existing checkpoint if available
# ═════════════════════════════════════════════════════════════════════════════

def load_or_train_v4(args, device, split_file):
    """Load V4 from existing checkpoint/history, or train it."""
    existing_ckpt    = Path(args.existing_checkpoint)
    existing_history = Path(args.existing_history)
    v4_dir = Path(args.output_dir) / "v4_full"

    if existing_ckpt.exists() and existing_history.exists() and not args.retrain_v4:
        print(f"\n{'='*60}")
        print(f"  V4: Full (proposed)  |  Loading existing checkpoint")
        print(f"{'='*60}")

        # Copy checkpoint to ablation dir
        v4_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(existing_ckpt, v4_dir / "best_model.pth")
        shutil.copy2(existing_history, v4_dir / "history.json")

        with open(existing_history) as f:
            history = json.load(f)

        # If test_metrics already saved in history, use them
        if "test_metrics" in history:
            print("  Test metrics loaded from existing history.")
            return history

        # Otherwise re-evaluate on test set
        print("  Re-evaluating on test set...")
        _, _, test_loader, _ = _make_loaders(
            args.data_dir, split_file,
            VARIANTS["v4_full"]["builder"],
            args.batch_size, img_size=256
        )
        model = UNetMini(in_channels=9, out_channels=1)
        ckpt  = torch.load(v4_dir / "best_model.pth",
                           map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        summary = evaluate(model, test_loader,
                           device=str(device), threshold=0.5,
                           results_dir=str(v4_dir / "test_results"))
        history["test_metrics"] = summary
        with open(v4_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        return history

    # No existing checkpoint — train from scratch
    return train_variant("v4_full", VARIANTS["v4_full"], args, device, split_file)


# ═════════════════════════════════════════════════════════════════════════════
# Summary table
# ═════════════════════════════════════════════════════════════════════════════

def _extract_metric(history, metric):
    """Safely extract a mean metric from test_metrics."""
    tm = history.get("test_metrics", {})
    entry = tm.get(metric, {})
    if isinstance(entry, dict):
        return entry.get("mean", float("nan"))
    return float("nan")


def build_summary(all_results):
    summary = {}
    for vname, history in all_results.items():
        label = VARIANTS[vname]["label"]
        summary[vname] = {
            "label":       label,
            "in_channels": VARIANTS[vname]["in_channels"],
            "best_epoch":  history.get("best_epoch", "—"),
            "val_dice":    history.get("best_val_dice", float("nan")),
            "test": {
                "dice":      _extract_metric(history, "dice"),
                "iou":       _extract_metric(history, "iou"),
                "recall":    _extract_metric(history, "recall"),
                "precision": _extract_metric(history, "precision"),
                "f2":        _extract_metric(history, "f2"),
            }
        }
    return summary


def print_table(summary, output_dir):
    lines = []
    header = (f"{'Variant':<32} {'Ch':>3} {'BEp':>4} "
              f"{'Dice':>7} {'IoU':>7} {'Recall':>7} "
              f"{'Prec.':>7} {'F2':>7}")
    sep = "-" * len(header)
    lines += [sep, header, sep]

    for vname in ["v1_baseline", "v2_clahe", "v3_features", "v4_full"]:
        if vname not in summary:
            continue
        s = summary[vname]
        t = s["test"]
        lines.append(
            f"{s['label']:<32} {s['in_channels']:>3} {s['best_epoch']:>4} "
            f"{t['dice']:>7.4f} {t['iou']:>7.4f} {t['recall']:>7.4f} "
            f"{t['precision']:>7.4f} {t['f2']:>7.4f}"
        )
    lines.append(sep)

    table_str = "\n".join(lines)
    print("\n\n" + table_str)

    table_path = Path(output_dir) / "ablation_table.txt"
    with open(table_path, "w") as f:
        f.write(table_str + "\n")
    print(f"\nTable saved → {table_path}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Ablation study for Kvasir-SEG polyp segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--data_dir",   default="Kvasir-SEG")
    p.add_argument("--split_file", default="split_index.json")
    p.add_argument("--output_dir", default="ablation_results")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--patience",   type=int,   default=15)
    p.add_argument("--device",     default=None)

    # V4 reuse options
    p.add_argument("--existing_checkpoint",
                   default="checkpoints/best_model.pth",
                   help="Existing V4 checkpoint to reuse (skip retraining)")
    p.add_argument("--existing_history",
                   default="checkpoints/history.json",
                   help="Existing V4 history.json to reuse")
    p.add_argument("--retrain_v4", action="store_true",
                   help="Force retrain V4 even if checkpoint exists")

    # Skip flags (useful if one variant already finished)
    p.add_argument("--skip_v1", action="store_true")
    p.add_argument("--skip_v2", action="store_true")
    p.add_argument("--skip_v3", action="store_true")
    p.add_argument("--skip_v4", action="store_true")

    return p.parse_args()


def main():
    args = _parse_args()

    if args.device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  |  epochs={args.epochs}  "
          f"|  output={args.output_dir}")

    all_results = {}

    # ── V1 ────────────────────────────────────────────────────────────
    if not args.skip_v1:
        all_results["v1_baseline"] = train_variant(
            "v1_baseline", VARIANTS["v1_baseline"],
            args, device, args.split_file
        )
    else:
        h = Path(args.output_dir) / "v1_baseline" / "history.json"
        if h.exists():
            with open(h) as f:
                all_results["v1_baseline"] = json.load(f)
            print("V1: loaded from existing results.")

    # ── V2 ────────────────────────────────────────────────────────────
    if not args.skip_v2:
        all_results["v2_clahe"] = train_variant(
            "v2_clahe", VARIANTS["v2_clahe"],
            args, device, args.split_file
        )
    else:
        h = Path(args.output_dir) / "v2_clahe" / "history.json"
        if h.exists():
            with open(h) as f:
                all_results["v2_clahe"] = json.load(f)
            print("V2: loaded from existing results.")

    # ── V3 ────────────────────────────────────────────────────────────
    if not args.skip_v3:
        all_results["v3_features"] = train_variant(
            "v3_features", VARIANTS["v3_features"],
            args, device, args.split_file
        )
    else:
        h = Path(args.output_dir) / "v3_features" / "history.json"
        if h.exists():
            with open(h) as f:
                all_results["v3_features"] = json.load(f)
            print("V3: loaded from existing results.")

    # ── V4 ────────────────────────────────────────────────────────────
    if not args.skip_v4:
        all_results["v4_full"] = load_or_train_v4(
            args, device, args.split_file
        )
    else:
        h = Path(args.output_dir) / "v4_full" / "history.json"
        if h.exists():
            with open(h) as f:
                all_results["v4_full"] = json.load(f)
            print("V4: loaded from existing results.")

    # ── Summary ───────────────────────────────────────────────────────
    summary = build_summary(all_results)
    summary_path = Path(args.output_dir) / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")

    print_table(summary, args.output_dir)
    print("\nDone. Use ablation_summary.json to update your LaTeX table.")


if __name__ == "__main__":
    main()