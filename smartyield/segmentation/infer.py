"""Vegetation segmentation at inference time.

Three methods are available, and the app records which one produced each mask:

* ``unet``      : trained CNN on RGB (works for any drone photo).
* ``ndvi_otsu`` : NDVI thresholded with Otsu's method (multispectral only).
                  This is the physically grounded choice when NIR is available.
* ``exg_otsu``  : Excess-Green thresholded with Otsu (classic RGB baseline).
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import torch

from .. import config, indices
from ..preprocessing import normalize_for_cnn
from .unet import UNet

UNET_PATH = config.MODELS_DIR / "unet_vegetation.pt"


def otsu_threshold(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi - lo < 1e-6:
        return float(lo)
    q = np.clip((v - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(q, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(lo + t / 255 * (hi - lo))


def exg_otsu_mask(rgb: np.ndarray) -> np.ndarray:
    exg = indices.rgb_indices(rgb)["exg"]
    t = otsu_threshold(exg)
    return np.nan_to_num(exg, nan=-9) > t


def ndvi_otsu_mask(ms: np.ndarray, floor: float = 0.3) -> np.ndarray:
    """NDVI > max(Otsu threshold, floor). The floor stops a fully vegetated
    image from being split into two "vegetation" halves."""
    nd = indices.multispectral_indices(ms, config.MAIA_BANDS)["ndvi"]
    t = max(otsu_threshold(nd), floor)
    v = nd[np.isfinite(nd)]
    if v.size and np.percentile(v, 10) > floor:      # canopy closed everywhere
        t = floor
    return np.nan_to_num(nd, nan=-9) > t


@lru_cache(maxsize=1)
def load_unet() -> tuple[UNet, dict] | None:
    if not UNET_PATH.exists():
        return None
    ck = torch.load(UNET_PATH, map_location="cpu", weights_only=False)
    m = UNet(widths=tuple(ck["widths"]))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck.get("meta", {})


@torch.no_grad()
def unet_probability(rgb: np.ndarray, tile: int = 512, overlap: int = 64) -> np.ndarray:
    """Sliding-window inference with overlap blending, for any image size."""
    loaded = load_unet()
    if loaded is None:
        raise FileNotFoundError(f"U-Net weights not found at {UNET_PATH}; run scripts/03_train_segmentation.py")
    model, meta = loaded
    scale = float(meta.get("inference_scale", 1.0))
    h0, w0 = rgb.shape[:2]
    img = cv2.resize(rgb, (max(32, int(w0 * scale)), max(32, int(h0 * scale))), interpolation=cv2.INTER_AREA) \
        if scale != 1.0 else rgb
    h, w = img.shape[:2]
    ph, pw = (-h) % 32, (-w) % 32
    img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    H, W = img.shape[:2]
    x = torch.from_numpy(normalize_for_cnn(img))[None]
    prob = np.zeros((H, W), np.float32)
    weight = np.zeros((H, W), np.float32)
    overlap = min(overlap, tile // 4)          # small tiles: keep a positive stride
    step = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)
    for yy in ys:
        for xx in xs:
            patch = x[..., yy:yy + tile, xx:xx + tile]
            p = torch.sigmoid(model(patch))[0, 0].numpy()
            prob[yy:yy + p.shape[0], xx:xx + p.shape[1]] += p
            weight[yy:yy + p.shape[0], xx:xx + p.shape[1]] += 1
    prob = (prob / np.maximum(weight, 1))[:h, :w]
    if prob.shape != (h0, w0):
        prob = cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return prob


def segment(rgb: np.ndarray, ms: np.ndarray | None = None, method: str = "auto") -> tuple[np.ndarray, str]:
    """Return (boolean vegetation mask, method used)."""
    if method == "auto":
        if ms is not None:
            method = "ndvi_otsu"
        else:
            method = config.RGB_SEG_METHOD if (config.RGB_SEG_METHOD != "unet" or UNET_PATH.exists()) else "exg_otsu"
    if method == "ndvi_otsu":
        if ms is None:
            raise ValueError("ndvi_otsu requires multispectral input")
        m = ndvi_otsu_mask(ms)
    elif method == "unet":
        m = unet_probability(rgb) > 0.5
    elif method == "exg_otsu":
        m = exg_otsu_mask(rgb)
    else:
        raise ValueError(method)
    # light morphological clean-up: remove isolated speckle
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    return m, method


def seg_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = float(np.sum(pred & gt))
    fp = float(np.sum(pred & ~gt))
    fn = float(np.sum(~pred & gt))
    tn = float(np.sum(~pred & ~gt))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "iou": tp / (tp + fp + fn + 1e-9),
        "dice": 2 * tp / (2 * tp + fp + fn + 1e-9),
        "precision": tp / (tp + fp + 1e-9),
        "recall": tp / (tp + fn + 1e-9),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + 1e-9),
    }
