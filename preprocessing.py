import cv2
import numpy as np

IMG_SIZE = 256


def resize(image, mask=None, size=IMG_SIZE):
    img_r = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    if mask is None:
        return img_r
    return img_r, cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def apply_clahe(image_bgr, clip_limit=2.0, tile_grid=(8, 8)):
    lab       = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b   = cv2.split(lab)
    clahe     = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab_eq    = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def normalize(image):
    return image.astype(np.float32) / 255.0


def preprocess_image(image_bgr_or_path, mask=None, size=IMG_SIZE):
    if isinstance(image_bgr_or_path, (str, bytes)):
        img = cv2.imread(str(image_bgr_or_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_bgr_or_path}")
        img = resize(img, None, size)
        img = apply_clahe(img)
        return normalize(img)          # returns bgr_float (H,W,3)

    image_bgr = image_bgr_or_path
    if mask is not None:
        image_bgr, mask = resize(image_bgr, mask, size)
    else:
        image_bgr = resize(image_bgr, None, size)

    image_f = normalize(apply_clahe(image_bgr))

    if mask is not None:
        return image_f, (mask > 127).astype(np.float32)
    return image_f, None