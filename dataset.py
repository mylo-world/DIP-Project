import os
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

BASE_DIR    = "Kvasir-SEG"
IMAGE_DIR   = os.path.join(BASE_DIR, "images")
MASK_DIR    = os.path.join(BASE_DIR, "masks")
BBOX_JSON   = os.path.join(BASE_DIR, "kavsir_bboxes.json")
OUTPUT_JSON = "split_index.json"


def load_dataset_paths(image_dir, mask_dir):
    image_paths, mask_paths, image_ids = [], [], []

    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith('.jpg'):
            continue

        img_path  = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        if not os.path.exists(mask_path):
            print(f"[WARNING] Mask not found for: {fname}, skipping.")
            continue

        image_paths.append(img_path)
        mask_paths.append(mask_path)
        image_ids.append(os.path.splitext(fname)[0])

    print(f"[INFO] Total images found: {len(image_paths)}")
    return image_paths, mask_paths, image_ids


def load_bbox_annotations(bbox_json_path, image_ids):
    with open(bbox_json_path, 'r') as f:
        raw = json.load(f)

    bbox_dict = {}
    for img_id in image_ids:
        if img_id in raw:
            bbox_dict[img_id] = raw[img_id]
        else:
            bbox_dict[img_id] = {"height": None, "width": None, "bbox": []}
            print(f"[WARNING] No bbox found for: {img_id}")

    found = sum(1 for v in bbox_dict.values() if len(v["bbox"]) > 0)
    print(f"[INFO] Bbox loaded: {found}/{len(image_ids)} images")
    return bbox_dict


def compute_polyp_size_ratios(mask_paths):
    ratios = []
    for mask_path in mask_paths:
        mask   = np.array(Image.open(mask_path).convert('L'))
        binary = (mask > 127).astype(np.float32)
        ratios.append(binary.mean())

    ratios = np.array(ratios)
    print(f"[INFO] Polyp size ratio — min: {ratios.min():.4f}, max: {ratios.max():.4f}, mean: {ratios.mean():.4f}")
    return ratios


def make_stratify_labels(ratios):
    # 0 = small (< 0.05), 1 = medium (0.05–0.30), 2 = large (>= 0.30)
    labels = np.zeros(len(ratios), dtype=int)
    labels[ratios >= 0.05] = 1
    labels[ratios >= 0.30] = 2

    names = {0: "small", 1: "medium", 2: "large"}
    for u, c in zip(*np.unique(labels, return_counts=True)):
        print(f"[INFO] {names[u]}: {c} images")

    return labels


def stratified_split(image_ids, labels, val_ratio=0.10, test_ratio=0.10, seed=42):
    indices = np.arange(len(image_ids))

    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, stratify=labels, random_state=seed
    )

    # val proportion relative to the remaining train_val pool
    val_size_adjusted = val_ratio / (1.0 - test_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size_adjusted,
        stratify=labels[train_val_idx], random_state=seed
    )

    train_ids = [image_ids[i] for i in train_idx]
    val_ids   = [image_ids[i] for i in val_idx]
    test_ids  = [image_ids[i] for i in test_idx]

    print(f"[INFO] Split — Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    return train_ids, val_ids, test_ids


def save_split_index(train_ids, val_ids, test_ids, output_path):
    with open(output_path, 'w') as f:
        json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2)
    print(f"[INFO] Split index saved to: {output_path}")


def main():
    image_paths, mask_paths, image_ids = load_dataset_paths(IMAGE_DIR, MASK_DIR)
    bbox_dict                          = load_bbox_annotations(BBOX_JSON, image_ids)
    ratios                             = compute_polyp_size_ratios(mask_paths)
    labels                             = make_stratify_labels(ratios)
    train_ids, val_ids, test_ids       = stratified_split(image_ids, labels)
    save_split_index(train_ids, val_ids, test_ids, OUTPUT_JSON)

    return {
        "image_paths": image_paths,
        "mask_paths":  mask_paths,
        "image_ids":   image_ids,
        "bbox_dict":   bbox_dict,
        "train_ids":   train_ids,
        "val_ids":     val_ids,
        "test_ids":    test_ids,
    }


if __name__ == "__main__":
    main()