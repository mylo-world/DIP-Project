import cv2
import numpy as np

def _warp_tensor(tensor, M, size=256, flags=cv2.INTER_LINEAR):
    out = np.zeros_like(tensor)
    for c in range(tensor.shape[2]):
        out[:, :, c] = cv2.warpAffine(
            tensor[:, :, c], M, (size, size),
            flags=flags, borderMode=cv2.BORDER_REFLECT_101
        )
    return out

def _warp_mask(mask, M, size=256):
    return cv2.warpAffine(
        mask, M, (size, size),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101
    )

def _horizontal_flip(tensor, mask, p=0.5, rng=None):
    rng = rng or np.random.default_rng()
    if rng.random() < p:
        tensor = tensor[:, ::-1, :].copy()
        mask   = mask[:, ::-1].copy()
    return tensor, mask

def _vertical_flip(tensor, mask, p=0.5, rng=None):
    rng = rng or np.random.default_rng()
    if rng.random() < p:
        tensor = tensor[::-1, :, :].copy()
        mask   = mask[::-1, :].copy()
    return tensor, mask

def _random_rotation(tensor, mask, max_angle=30, rng=None):
    rng   = rng or np.random.default_rng()
    angle = rng.uniform(-max_angle, max_angle)
    size  = tensor.shape[0]
    cx, cy = size // 2, size // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    return _warp_tensor(tensor, M, size), _warp_mask(mask, M, size)

def _random_zoom(tensor, mask, scale_range=(0.8, 1.2), rng=None):
    rng   = rng or np.random.default_rng()
    scale = rng.uniform(*scale_range)
    size  = tensor.shape[0]
    cx, cy = size // 2, size // 2
    M = cv2.getRotationMatrix2D((cx, cy), 0.0, scale)
    return _warp_tensor(tensor, M, size), _warp_mask(mask, M, size)

def _random_crop(tensor, mask, crop_size=224, out_size=256, rng=None):
    rng = rng or np.random.default_rng()
    h, w = tensor.shape[:2]

    x = int(rng.integers(0, w - crop_size + 1))
    y = int(rng.integers(0, h - crop_size + 1))

    tensor_cropped = tensor[y:y+crop_size, x:x+crop_size, :]
    mask_cropped   = mask[y:y+crop_size, x:x+crop_size]

    tensor_resized = np.stack([
        cv2.resize(tensor_cropped[:, :, c], (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        for c in range(tensor_cropped.shape[2])
    ], axis=2).astype(np.float32)

    mask_resized = cv2.resize(mask_cropped, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return tensor_resized, mask_resized

def _random_shear(tensor, mask, max_shear_deg=15, rng=None):
    rng       = rng or np.random.default_rng()
    shear_rad = np.deg2rad(rng.uniform(-max_shear_deg, max_shear_deg))
    size      = tensor.shape[0]
    M = np.float32([
        [1, np.tan(shear_rad), 0],
        [0, 1,                 0],
    ])
    return _warp_tensor(tensor, M, size), _warp_mask(mask, M, size)

def augment(tensor: np.ndarray, mask: np.ndarray, seed: int = None) -> tuple:
    """
    Apply synchronized spatial augmentation to tensor and mask.
    Augmentation is only applied during training (not val/test).

    Both tensor (H,W,9) and mask (H,W) are transformed identically
    to preserve spatial alignment between input and ground truth.
    """
    rng = np.random.default_rng(seed)

    tensor, mask = _horizontal_flip(tensor, mask, p=0.5,  rng=rng)
    tensor, mask = _vertical_flip(tensor,   mask, p=0.5,  rng=rng)
    tensor, mask = _random_rotation(tensor, mask, max_angle=30,          rng=rng)
    tensor, mask = _random_zoom(tensor,     mask, scale_range=(0.8, 1.2), rng=rng)
    tensor, mask = _random_crop(tensor,     mask, crop_size=224, out_size=256, rng=rng)
    tensor, mask = _random_shear(tensor,    mask, max_shear_deg=15,       rng=rng)

    tensor = np.clip(tensor, 0.0, 1.0)
    mask   = (mask > 0.5).astype(np.float32)
    return tensor, mask



if __name__ == "__main__":
    print("[TEST] augmentation.py\n")

    dummy_tensor = np.random.rand(256, 256, 9).astype(np.float32)
    dummy_mask   = (np.random.rand(256, 256) > 0.7).astype(np.float32)

    aug_tensor, aug_mask = augment(dummy_tensor, dummy_mask, seed=0)

    assert aug_tensor.shape == (256, 256, 9), f"Wrong tensor shape: {aug_tensor.shape}"
    assert aug_mask.shape   == (256, 256),    f"Wrong mask shape: {aug_mask.shape}"
    assert aug_tensor.min() >= 0.0 and aug_tensor.max() <= 1.0, "Tensor out of range [0, 1]!"
    assert set(np.unique(aug_mask)).issubset({0.0, 1.0}), "Mask is not binary!"

    print(f"  tensor : {aug_tensor.shape}  range [{aug_tensor.min():.3f}, {aug_tensor.max():.3f}]")
    print(f"  mask   : {aug_mask.shape}   unique={np.unique(aug_mask)}")
    print("\n[OK] All assertions passed.")