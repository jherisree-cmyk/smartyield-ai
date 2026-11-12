"""Vegetation indices.

Two families, kept strictly separate:

* Multispectral indices (need a near-infrared and/or red-edge band): NDVI, NDRE,
  GNDVI, OSAVI and CIre. These come only from multispectral imagery such as the
  9-band MAIA camera.
* RGB indices (visible bands only): ExG, ExGR, VARI, GLI, NGRDI and TGI. These
  are colour-based greenness proxies. They are NOT NDVI and must never be shown
  as NDVI.

All functions accept float arrays. Multispectral inputs are surface
reflectance (0..1). RGB inputs are any non-negative scale, and the chromatic
coordinates make most RGB indices insensitive to overall brightness.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6

MS_INDICES = ("ndvi", "ndre", "gndvi", "osavi", "cire")
RGB_INDICES = ("exg", "exgr", "vari", "gli", "ngrdi", "tgi")


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float32)
    ok = np.abs(b) > EPS
    out[ok] = a[ok] / b[ok]
    return out


# ---------------------------------------------------------------- multispectral
def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red). Range -1..1, dense green canopy ~0.7-0.9."""
    return _safe_div(nir - red, nir + red)


def ndre(nir: np.ndarray, rededge: np.ndarray) -> np.ndarray:
    """Normalised Difference Red Edge. Saturates later than NDVI in dense canopy."""
    return _safe_div(nir - rededge, nir + rededge)


def gndvi(nir: np.ndarray, green: np.ndarray) -> np.ndarray:
    return _safe_div(nir - green, nir + green)


def osavi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Optimised Soil-Adjusted VI. Reduces the soil/water background effect in sparse canopy."""
    return _safe_div(1.16 * (nir - red), nir + red + 0.16)


def cire(nir: np.ndarray, rededge: np.ndarray) -> np.ndarray:
    """Red-edge chlorophyll index, related to canopy chlorophyll and N status."""
    return _safe_div(nir, rededge) - 1.0


def multispectral_indices(ms: np.ndarray, bands: dict[str, int]) -> dict[str, np.ndarray]:
    """Compute all multispectral indices from an HxWxB reflectance cube."""
    b = {k: ms[..., i].astype(np.float32) for k, i in bands.items() if i < ms.shape[-1]}
    return {
        "ndvi": ndvi(b["nir"], b["red"]),
        "ndre": ndre(b["nir"], b["re1"]),
        "gndvi": gndvi(b["nir"], b["green"]),
        "osavi": osavi(b["nir"], b["red"]),
        "cire": cire(b["nir"], b["re1"]),
    }


# ------------------------------------------------------------------------- RGB
def chromatic(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chromatic coordinates r, g, b, each channel divided by R+G+B."""
    rgb = rgb.astype(np.float32)
    s = rgb.sum(axis=-1)
    s[s < EPS] = np.nan
    return rgb[..., 0] / s, rgb[..., 1] / s, rgb[..., 2] / s


def rgb_indices(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Visible-band greenness indices from an HxWx3 RGB array (any scale)."""
    x = rgb.astype(np.float32)
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    r, g, b = chromatic(x)
    exg = 2 * g - r - b                          # Excess Green (Woebbecke 1995)
    exr = 1.4 * r - g                            # Excess Red
    return {
        "exg": exg,
        "exgr": exg - exr,                       # ExG - ExR (Meyer & Neto 2008)
        "vari": _safe_div(G - R, G + R - B),     # Visible Atmospherically Resistant Index
        "gli": _safe_div(2 * G - R - B, 2 * G + R + B),  # Green Leaf Index
        "ngrdi": _safe_div(G - R, G + R),        # Normalised Green-Red Difference
        "tgi": -0.5 * (190 * (r - g) - 120 * (r - b)),   # Triangular Greenness (chromatic form)
    }


def clean(a: np.ndarray, lo: float = -1.5, hi: float = 1.5) -> np.ndarray:
    """Clip extreme values (division blow-ups) and keep NaNs as missing."""
    a = a.astype(np.float32)
    return np.clip(a, lo, hi, where=np.isfinite(a), out=a)
