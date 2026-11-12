"""Crop health classification.

Multispectral images: true NDVI, classified *relative to the image*: crop
pixels in the bottom 20 % of the image's NDVI distribution = poor, middle =
moderate, top 40 % = healthy. Absolute NDVI is still reported. Absolute class
breaks were tested (textbook and data-calibrated). Experiment 3 showed that
they mainly separate sensors/seasons, not crop condition, so they are not used.
RGB images: there is no NIR, so absolute NDVI-style classes are impossible. A
*relative* greenness index (VARI) is used instead. Vegetation pixels are
ranked within the image (bottom 20 % = poor, middle = moderate, top 40 % =
healthy). The app labels this "relative RGB greenness", never NDVI.
"""
from __future__ import annotations

import numpy as np

from . import config, indices

CLASSES = ["background", "poor", "moderate", "healthy"]
COLORS = np.array([[120, 120, 120], [208, 59, 59], [250, 178, 25], [12, 163, 12]], np.uint8)
DEFAULT_NDVI_BREAKS = (0.30, 0.55, 0.75)        # textbook values, used only if no calibration exists
CALIBRATION = config.MODELS_DIR / "health_calibration.json"


def ndvi_breaks() -> tuple[tuple[float, float, float], str]:
    """NDVI class breaks. They are calibrated on the training imagery when
    available, because the absolute NDVI level depends on the sensor and its
    radiometric processing (see docs/DATASETS.md, NDVI validation)."""
    if CALIBRATION.exists():
        import json

        c = json.loads(CALIBRATION.read_text())
        return (c["background_below"], c["poor_below"], c["healthy_from"]), "NDVI classes calibrated on training imagery (20th/60th percentiles of plot NDVI)"
    return DEFAULT_NDVI_BREAKS, "textbook absolute NDVI thresholds"


def health_map(rgb: np.ndarray, veg: np.ndarray, ms: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Return (class map HxW uint8 0..3, info dict)."""
    cls = np.zeros(veg.shape, np.uint8)
    if ms is not None:
        nd = indices.multispectral_indices(ms, config.MAIA_BANDS)["ndvi"]
        nd = np.nan_to_num(nd, nan=-1)
        v = nd[veg]
        # Relative within-image classes. Experiment 3 showed that absolute NDVI breaks (even
        # data-calibrated ones) mostly separate sensor/processing seasons, not crop condition.
        p20, p60 = (np.percentile(v, [20, 60]) if v.size else (None, None))
        if v.size:
            cls[veg] = 1
            cls[veg & (nd >= p20)] = 2
            cls[veg & (nd >= p60)] = 3
        info = {"index": "NDVI", "basis": "relative within-image NDVI percentiles (20/60) of crop pixels; "
                "true NDVI value reported", "breaks": [p20, p60],
                "index_mean_veg": float(v.mean()) if v.size else None}
        idx = nd
    else:
        vari = np.nan_to_num(indices.clean(indices.rgb_indices(rgb)["vari"], -1, 1), nan=-1)
        v = vari[veg]
        if v.size:
            p20, p60 = np.percentile(v, [20, 60])
            cls[veg] = 1
            cls[veg & (vari >= p20)] = 2
            cls[veg & (vari >= p60)] = 3
        else:
            p20 = p60 = None
        info = {"index": "VARI (RGB)", "basis": "relative within-image percentiles (20/60) - NOT NDVI",
                "breaks": [p20, p60], "index_mean_veg": float(v.mean()) if v.size else None}
        idx = vari
    n = cls.size
    info["fractions"] = {c: float((cls == i).sum() / n) for i, c in enumerate(CLASSES)}
    info["index_map"] = idx
    return cls, info


def colorize(cls: np.ndarray) -> np.ndarray:
    return COLORS[cls]
