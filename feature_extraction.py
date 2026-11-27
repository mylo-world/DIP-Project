import cv2
import numpy as np
from skimage.feature import local_binary_pattern


# extract LBP texture map
def extract_lbp(bgr_float):
    gray = cv2.cvtColor((bgr_float * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # radius=3, n_points=24, method='uniform'
    lbp     = local_binary_pattern(gray, P=24, R=3, method='uniform')

    # normalize to [0, 1]
    max_val = lbp.max()
    return (lbp / max_val if max_val > 0 else lbp).astype(np.float32)


# extract Canny edge map
def extract_canny(bgr_float):
    gray  = cv2.cvtColor((bgr_float * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # low=50, high=150 as per proposal
    canny = cv2.Canny(gray, threshold1=50, threshold2=150)

    return (canny / 255.0).astype(np.float32)


# extract HSV channels
def extract_hsv(bgr_float):
    hsv = cv2.cvtColor((bgr_float * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)

    h = (hsv[:, :, 0] / 179.0).astype(np.float32)
    s = (hsv[:, :, 1] / 255.0).astype(np.float32)
    v = (hsv[:, :, 2] / 255.0).astype(np.float32)

    return h, s, v


# build BBox spatial prior map
def build_bbox_map(bbox_entry, target_size=256):
    # bbox_entry: { "height": H, "width": W, "bbox": [{"label": "polyp", "xmin": x, "ymin": y, "xmax": x, "ymax": y}, ...] }
    bbox_map = np.zeros((target_size, target_size), dtype=np.float32)

    orig_w = bbox_entry.get("width")
    orig_h = bbox_entry.get("height")

    if not orig_w or not orig_h:
        return bbox_map

    for box in bbox_entry.get("bbox", []):
        # scale from original resolution to target_size
        x_min = int(box["xmin"] / orig_w * target_size)
        y_min = int(box["ymin"] / orig_h * target_size)
        x_max = int(box["xmax"] / orig_w * target_size)
        y_max = int(box["ymax"] / orig_h * target_size)

        # clamp to stay within [0, target_size]
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(target_size, x_max), min(target_size, y_max)

        bbox_map[y_min:y_max, x_min:x_max] = 1.0

    return bbox_map


# build 9-channel input tensor for a single image
def build_tensor(bgr_float, bbox_entry):
    # convert BGR -> RGB for the 3 base channels
    rgb = bgr_float[:, :, ::-1]

    lbp      = extract_lbp(bgr_float)
    canny    = extract_canny(bgr_float)
    h, s, v  = extract_hsv(bgr_float)
    bbox_map = build_bbox_map(bbox_entry)

    # channel order: RGB(3) + LBP(1) + Canny(1) + H(1) + S(1) + V(1) + BBox(1) = 9
    tensor = np.concatenate([
        rgb,
        lbp[:, :, np.newaxis],
        canny[:, :, np.newaxis],
        h[:, :, np.newaxis],
        s[:, :, np.newaxis],
        v[:, :, np.newaxis],
        bbox_map[:, :, np.newaxis],
    ], axis=2)

    return tensor.astype(np.float32)


if __name__ == "__main__":
    import os
    import json
    from preprocessing import preprocess_image

    BASE_DIR   = "Kvasir-SEG"
    IMAGE_DIR  = os.path.join(BASE_DIR, "images")
    SPLIT_JSON = "split_index.json"
    BBOX_JSON  = os.path.join(BASE_DIR, "kavsir_bboxes.json")

    with open(SPLIT_JSON, 'r') as f:
        split = json.load(f)
    with open(BBOX_JSON, 'r') as f:
        bbox_data = json.load(f)

    sample_ids = split["train"][:3]

    for img_id in sample_ids:
        image_path = os.path.join(IMAGE_DIR, img_id + ".jpg")
        bgr_float  = preprocess_image(image_path)
        bbox_entry = bbox_data.get(img_id, {"height": None, "width": None, "bbox": []})
        tensor     = build_tensor(bgr_float, bbox_entry)

        print(f"[{img_id}] tensor shape: {tensor.shape} | "
              f"range: [{tensor.min():.2f}, {tensor.max():.2f}] | "
              f"dtype: {tensor.dtype} | "
              f"bbox filled: {(tensor[:,:,8] > 0).sum()} px")

    print("[INFO] Feature extraction check complete.")
