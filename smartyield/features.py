"""Feature extraction: turn the pixels of one region (a plot, a grid cell or a
whole image) into a fixed-length numeric vector.

`region_features` is the only feature function in the project. The training
dataset (scripts/02_build_plot_dataset.py) and the live app
(smartyield/analysis.py) both call it, so no training/serving skew is possible.

Feature groups
--------------
coverage   : vegetation cover fraction, green-pixel %, canopy density, gap fraction
rgb_*      : visible-band index statistics (ExG, ExGR, VARI, GLI, NGRDI, TGI) + chromatic coords
ms_*       : NDVI / NDRE / GNDVI / OSAVI / CIre statistics + band reflectances (multispectral only)
tex_*      : GLCM texture (contrast, homogeneity, energy, correlation, entropy) + sharpness
context    : day of year of the flight (crop-development timing)
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config, indices

PCTS = (10, 50, 90)


def _stats(prefix: str, a: np.ndarray, sel: np.ndarray, full: bool = True) -> dict[str, float]:
    v = a[sel]
    v = v[np.isfinite(v)]
    if v.size == 0:
        out = {f"{prefix}_mean": np.nan}
        if full:
            out[f"{prefix}_std"] = np.nan
            out.update({f"{prefix}_p{p}": np.nan for p in PCTS})
        return out
    out = {f"{prefix}_mean": float(v.mean())}
    if full:
        out[f"{prefix}_std"] = float(v.std())
        out.update({f"{prefix}_p{p}": float(x) for p, x in zip(PCTS, np.percentile(v, PCTS))})
    return out


def glcm_features(gray: np.ndarray, sel: np.ndarray, levels: int = 16) -> dict[str, float]:
    """Grey-level co-occurrence matrix texture (Haralick), offsets of 1 px at 0 and 90 degrees.

    Texture describes canopy structure. A closed, uniform canopy is homogeneous
    with low contrast. Patchy growth with soil or water gaps has high contrast
    and lower energy.
    """
    g = gray.astype(np.float32)
    v = g[sel & np.isfinite(g)]
    if v.size < 16:
        return {k: np.nan for k in ("tex_contrast", "tex_homogeneity", "tex_energy", "tex_correlation", "tex_entropy")}
    lo, hi = np.percentile(v, [1, 99])
    q = np.clip(((g - lo) / max(hi - lo, 1e-6) * (levels - 1)).round(), 0, levels - 1)
    q = np.nan_to_num(q, nan=0).astype(np.int64)
    P = np.zeros((levels, levels), np.float64)
    for dy, dx in ((0, 1), (1, 0)):
        a = q[: q.shape[0] - dy, : q.shape[1] - dx]
        b = q[dy:, dx:]
        m = sel[: sel.shape[0] - dy, : sel.shape[1] - dx] & sel[dy:, dx:]
        if m.any():
            P += np.bincount(a[m] * levels + b[m], minlength=levels * levels).reshape(levels, levels)
    P = P + P.T
    if P.sum() == 0:
        return {k: np.nan for k in ("tex_contrast", "tex_homogeneity", "tex_energy", "tex_correlation", "tex_entropy")}
    P /= P.sum()
    i, j = np.indices(P.shape)
    mu_i, mu_j = (i * P).sum(), (j * P).sum()
    sd_i = np.sqrt(((i - mu_i) ** 2 * P).sum())
    sd_j = np.sqrt(((j - mu_j) ** 2 * P).sum())
    nz = P[P > 0]
    return {
        "tex_contrast": float(((i - j) ** 2 * P).sum()),
        "tex_homogeneity": float((P / (1 + np.abs(i - j))).sum()),
        "tex_energy": float(np.sqrt((P ** 2).sum())),
        "tex_correlation": float(((i - mu_i) * (j - mu_j) * P).sum() / max(sd_i * sd_j, 1e-9)),
        "tex_entropy": float(-(nz * np.log2(nz)).sum()),
    }


def region_features(rgb: np.ndarray, veg: np.ndarray, region: np.ndarray | None = None,
                    ms: np.ndarray | None = None, doy: float | None = None) -> dict[str, float]:
    """Features for one region.

    rgb    : HxWx3 uint8 (analysis image, no CLAHE)
    veg    : HxW bool vegetation mask (from segmentation)
    region : HxW bool region-of-interest (plot polygon / grid cell); None = whole image
    ms     : HxWxB float reflectance or None
    doy    : day of year of image capture
    """
    if region is None:
        region = np.ones(rgb.shape[:2], bool)
    valid = region & (rgb.sum(-1) > 0)
    if ms is not None:
        valid &= np.isfinite(ms).all(-1)
    n = int(valid.sum())
    f: dict[str, float] = {"n_pixels": float(n)}
    if n == 0:
        return f
    vsel = valid & veg

    # ---- coverage / canopy
    cover = vsel.sum() / n
    r, g, b = indices.chromatic(rgb)
    green_px = (g > r) & (g > b)
    dens = cv2.blur(veg.astype(np.float32), (7, 7))            # local vegetation density
    f.update({
        "veg_cover": float(cover),
        "gap_fraction": float(1 - cover),
        "green_pixel_pct": float(100 * (green_px & valid).sum() / n),
        "canopy_density": float(dens[valid].mean()),
    })

    # ---- RGB indices (all valid pixels + vegetation-only means)
    ri = {k: indices.clean(v) for k, v in indices.rgb_indices(rgb).items()}
    for k in indices.RGB_INDICES:
        f.update(_stats(f"rgb_{k}", ri[k], valid, full=k in ("exg", "vari", "gli")))
        f.update({f"rgb_{k}_veg_mean": _stats("x", ri[k], vsel, full=False)["x_mean"]})
    f.update({"rgb_r_chrom": float(np.nanmean(r[valid])), "rgb_g_chrom": float(np.nanmean(g[valid])),
              "rgb_b_chrom": float(np.nanmean(b[valid]))})

    # ---- texture
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    f.update(glcm_features(gray, valid))
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    f["tex_laplacian_var"] = float(lap[valid].var())

    # ---- multispectral
    if ms is not None:
        mi = {k: indices.clean(v) for k, v in indices.multispectral_indices(ms, config.MAIA_BANDS).items()}
        for k in indices.MS_INDICES:
            f.update(_stats(f"ms_{k}", mi[k], valid, full=k in ("ndvi", "ndre", "cire")))
            f[f"ms_{k}_veg_mean"] = _stats("x", mi[k], vsel, full=False)["x_mean"]
        for bn in ("blue", "green", "red", "re1", "re2", "re3", "nir"):
            f[f"ms_refl_{bn}"] = float(np.nanmean(ms[..., config.MAIA_BANDS[bn]][valid]))
        ndvi_tex = glcm_features(np.nan_to_num(mi["ndvi"], nan=0), valid)
        f["ms_ndvi_tex_contrast"] = ndvi_tex["tex_contrast"]
        f["ms_ndvi_tex_homogeneity"] = ndvi_tex["tex_homogeneity"]

    if doy is not None:
        f["doy"] = float(doy)
    return f


def feature_columns(track: str, all_cols) -> list[str]:
    """Model input columns for a track: 'rgb' (any camera) or 'ms' (multispectral)."""
    base = [c for c in all_cols if c in ("veg_cover", "gap_fraction", "green_pixel_pct", "canopy_density", "doy")
            or c.startswith("rgb_") or c.startswith("tex_")]
    if track == "rgb":
        return base
    if track == "ms":
        return base + [c for c in all_cols if c.startswith("ms_")]
    raise ValueError(track)
