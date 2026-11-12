"""End-to-end analysis of one uploaded drone image. Used by the API and the CLI.

    load -> preprocess -> segment -> health map -> grid zones -> features per
    zone -> yield model per zone -> field aggregation -> visual products

Field-level aggregation (Module 7)
---------------------------------
The image is split into a NON-overlapping grid of square zones. When the
image is georeferenced, each zone is ~1 m x 1 m (similar in size to the
training plots, which were 0.3-0.5 m^2 harvest areas). Each zone gets its own
features and its own yield prediction (t/ha). Zones do not overlap, so each
square metre is counted exactly once:

    field yield (t/ha)  = area-weighted mean of zone predictions over cropped zones
    production (t)      = field yield x cropped area (ha)

Cropped area comes from georeferencing (GSD) when available. Otherwise the
user-entered field area is scaled by the image's cropped fraction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from . import config, health
from .features import region_features
from .preprocessing import DroneImage, load_image, preprocess_rgb
from .segmentation.infer import segment

MAX_SIDE_RGB = 2048


@lru_cache(maxsize=4)
def load_bundle(track: str):
    p = config.MODELS_DIR / f"yield_{track}.joblib"
    return joblib.load(p) if p.exists() else None


@dataclass
class AnalysisResult:
    summary: dict
    zones: pd.DataFrame
    images: dict          # name -> HxWx3 uint8


def _zone_grid(h: int, w: int, gsd_m: float | None, target_m: float = 1.0, n_default: int = 12):
    if gsd_m:
        cell = max(8, int(round(target_m / gsd_m)))
    else:
        cell = max(16, int(round(max(h, w) / n_default)))
    return cell, [(y, x) for y in range(0, h, cell) for x in range(0, w, cell)]


def analyze(path: str | Path, capture_date: date | None = None, field_area_ha: float | None = None,
            crop: str = "rice", seg_method: str = "auto") -> AnalysisResult:
    if crop not in config.CROPS:
        raise ValueError(f"crop '{crop}' is not supported yet; supported: {list(config.CROPS)}")
    img: DroneImage = load_image(path, max_side=None if str(path).lower().endswith((".tif", ".tiff")) else MAX_SIDE_RGB)
    if img.is_multispectral and max(img.shape) > 4096:      # very large orthomosaic: analyse at reduced GSD
        img = load_image(path, max_side=4096)
    track = "ms" if img.is_multispectral else "rgb"
    capture_date = capture_date or date.today()
    doy = capture_date.timetuple().tm_yday

    # ---- preprocessing (RGB photos only; calibrated reflectance is already corrected)
    if img.is_multispectral:
        pre = {"original": img.rgb, "analysis": img.rgb, "enhanced": img.rgb}
        ms = np.nan_to_num(img.ms, nan=0.0)
    else:
        pre = preprocess_rgb(img.rgb)
        ms = None
    rgb = pre["analysis"]
    valid = rgb.sum(-1) > 0

    # ---- segmentation + health
    veg, method = segment(rgb, ms, seg_method)
    veg &= valid
    cls, hinfo = health.health_map(rgb, veg, ms)

    # ---- ground sampling distance: from georeferencing, else estimated from the
    # user-entered field area (assumes the image footprint is the field, typical for a
    # nadir drone photo framed on one field), else unknown
    gsd_source = "georeferenced" if img.gsd_m else None
    if not img.gsd_m and field_area_ha and valid.any():
        img.gsd_m = float(np.sqrt(field_area_ha * 1e4 / valid.sum()))
        gsd_source = "estimated from field area"

    # ---- zones
    h, w = veg.shape
    cell, cells = _zone_grid(h, w, img.gsd_m)
    rows = []
    for (y0, x0) in cells:
        sl = (slice(y0, y0 + cell), slice(x0, x0 + cell))
        reg = valid[sl]
        if reg.sum() < 0.5 * cell * cell:
            continue
        f = region_features(rgb[sl], veg[sl], reg, ms=None if ms is None else ms[sl], doy=doy)
        f.update({"zone_y": y0, "zone_x": x0, "zone_px": int(reg.sum())})
        rows.append(f)
    zones = pd.DataFrame(rows)
    zones["is_crop"] = zones.veg_cover >= 0.25 if len(zones) else []
    trained = config.CROPS[crop]["trained"]
    bundle = load_bundle(track) if trained else None
    if trained and bundle is None:
        raise FileNotFoundError(f"No trained yield model for track '{track}'. Run scripts/05_train_yield.py")

    # ---- relative productivity index (all crops): zone greenness x cover relative to the
    # field's median crop zone (1.0 = typical for this field, 0.8 = 20 % below). It is unitless
    # and is NOT a yield; for untrained crops it is the only productivity output.
    gidx = zones["ms_ndvi_veg_mean"] if "ms_ndvi_veg_mean" in zones else zones["rgb_vari_veg_mean"].clip(lower=0) + 0.05
    vigour = gidx.clip(lower=0) * zones.veg_cover
    ref = float(np.nanmedian(vigour[zones.is_crop])) if zones.is_crop.any() else np.nan
    zones["rel_productivity"] = (vigour / ref).where(zones.is_crop) if ref and np.isfinite(ref) and ref > 0 else np.nan

    if bundle is not None:
        zones = pd.concat([zones, bundle.predict_frame(zones, crop_mask=zones.is_crop)], axis=1)
        # no yield is attributed to bare soil / water zones
        zones.loc[~zones.is_crop, ["pred_t_ha", "lo90_t_ha", "hi90_t_ha"]] = np.nan
        score = zones.pred_t_ha
    else:
        for c in ("pred_t_ha", "lo90_t_ha", "hi90_t_ha", "ood_fraction"):
            zones[c] = np.nan
        score = zones.rel_productivity
    low = np.nanpercentile(score[zones.is_crop], 20) if zones.is_crop.any() and score[zones.is_crop].notna().any() else -np.inf
    zones["poor_growth"] = zones.is_crop & ((zones.veg_cover < 0.6) | (score < low))

    # ---- areas + aggregation
    px_area_m2 = img.gsd_m ** 2 if img.gsd_m else None
    crop_frac = float(zones.loc[zones.is_crop, "zone_px"].sum() / max(zones.zone_px.sum(), 1))
    if gsd_source == "georeferenced":
        analysed_ha = zones.zone_px.sum() * px_area_m2 / 1e4
        area_source = "georeferenced GSD"
    elif field_area_ha:
        analysed_ha = field_area_ha
        area_source = "user-entered field area"
    else:
        analysed_ha = None
        area_source = "unknown (enter field area to get production)"
    cz = zones[zones.is_crop]
    if bundle is not None and len(cz) and cz.pred_t_ha.notna().any():
        wts = cz.zone_px.to_numpy()
        field_yield = float(np.average(cz.pred_t_ha, weights=wts))
        lo = float(np.average(cz.lo90_t_ha, weights=wts)) if "lo90_t_ha" in cz else None
        hi = float(np.average(cz.hi90_t_ha, weights=wts)) if "hi90_t_ha" in cz else None
    else:
        field_yield = lo = hi = None
    cropped_ha = analysed_ha * crop_frac if analysed_ha else None
    production_t = field_yield * cropped_ha if (field_yield is not None and cropped_ha) else None

    # ---- visual products
    images = {
        "original": img.rgb, "enhanced": pre["enhanced"],
        "segmentation": _overlay(img.rgb, veg),
        "mask": (veg * 255).astype(np.uint8)[..., None].repeat(3, -1),
        "health": health.colorize(cls),
        "yield_map": _yield_map(img.rgb, zones, cell, config.CROPS[crop]["typical_range_t_ha"],
                                "pred_t_ha" if bundle is not None else "rel_productivity"),
    }
    summary = {
        "crop": crop, "crop_label": config.CROPS[crop]["label"], "track": track,
        "yield_model_available": bundle is not None,
        "map_variable": "predicted yield (t/ha)" if bundle is not None else "relative productivity index (unitless)",
        "rel_productivity_stats": {
            "p10": float(np.nanpercentile(zones.rel_productivity, 10)) if zones.rel_productivity.notna().any() else None,
            "p90": float(np.nanpercentile(zones.rel_productivity, 90)) if zones.rel_productivity.notna().any() else None,
        },
        "track_label": "Multispectral (true NDVI available)" if track == "ms" else "RGB only (no NIR: NDVI not computed)",
        "segmentation_method": method, "image_size": [int(h), int(w)],
        "gsd_m": img.gsd_m, "gsd_source": gsd_source, "capture_date": capture_date.isoformat(), "doy": doy,
        "vegetation_cover_pct": float(100 * veg[valid].mean()) if valid.any() else 0.0,
        "green_pixel_pct": float(np.mean(zones.green_pixel_pct)) if len(zones) else 0.0,
        "canopy_density": float(np.mean(zones.canopy_density)) if len(zones) else 0.0,
        "health": {k: v for k, v in hinfo.items() if k != "index_map"},
        "n_zones": int(len(zones)), "n_crop_zones": int(zones.is_crop.sum()),
        "n_poor_growth_zones": int(zones.poor_growth.sum()), "zone_size_px": cell,
        "zone_size_m": cell * img.gsd_m if img.gsd_m else None,
        "predicted_yield_t_ha": field_yield, "yield_lo90_t_ha": lo, "yield_hi90_t_ha": hi,
        "predicted_yield_kg_acre": field_yield * 1000 / 2.47105 if field_yield is not None else None,
        "analysed_area_ha": analysed_ha, "cropped_area_ha": cropped_ha, "area_source": area_source,
        "estimated_production_t": production_t,
        "model": ({"name": bundle.model_name, "track": track, "cv_rmse_t_ha": bundle.cv_metrics["image_level"]["RMSE"],
                   "cv_r2": bundle.cv_metrics["image_level"]["R2"], "interval": "split-conformal 90 %",
                   "trained_on": bundle.training_summary.get("dataset")} if bundle is not None else
                  {"name": None, "track": track, "cv_rmse_t_ha": None, "cv_r2": None, "interval": None,
                   "trained_on": None}),
        "ood_warning": bool(np.nanmean(zones.ood_fraction) > 0.2) if (bundle is not None and len(zones)) else False,
        "ood_fraction_mean": float(np.nanmean(zones.ood_fraction)) if (bundle is not None and len(zones)) else None,
        "zone_yield_stats": {
            "min": float(cz.pred_t_ha.min()) if field_yield is not None else None,
            "max": float(cz.pred_t_ha.max()) if field_yield is not None else None,
            "sd": float(cz.pred_t_ha.std()) if field_yield is not None and len(cz) > 1 else None,
        },
    }
    summary["recommendations"], summary["scouting"] = _recommendations(zones, summary, img, cell)
    summary = _json_safe(summary)
    if bundle is None:
        summary["yield_note"] = (f"No validated yield model exists yet for {config.CROPS[crop]['label']}. A t/ha estimate needs "
                                 "a model trained on real harvest measurements of this crop, and SmartYield AI does not invent one. "
                                 "Shown instead: crop cover, health and a relative productivity map (1.0 = field median).")
    if track == "rgb":
        summary["notes"] = ("RGB-only image: vegetation indices are colour-based (ExG, VARI, GLI). "
                            "They are not NDVI. The RGB model was trained on RGB renderings of calibrated "
                            "multispectral imagery. Consumer-camera colours differ, so treat predictions as indicative.")
    return AnalysisResult(summary=summary, zones=zones, images=images)


def _recommendations(zones: pd.DataFrame, s: dict, img: DroneImage, cell: int):
    """Rule-based, conservative advisories + a scouting list of the worst zones.

    These are prompts for field inspection, not prescriptions: imagery alone
    cannot tell nitrogen deficiency from water stress, disease or a variety effect.
    """
    rec = []
    cz = zones[zones.is_crop]
    n_crop = max(len(cz), 1)
    gap = zones.is_crop & (zones.veg_cover < 0.6)
    weak = zones.poor_growth & ~gap
    if s["vegetation_cover_pct"] < 50:
        rec.append({"priority": "high", "title": "Low crop cover across the image",
                    "detail": f"Only {s['vegetation_cover_pct']:.0f} % of the image is crop canopy. Early growth stage, poor "
                              "establishment or a lot of bare soil/water in frame. Check stand density and gap-fill if needed."})
    if gap.sum() / n_crop > 0.1:
        rec.append({"priority": "high", "title": f"{int(gap.sum())} zones with patchy canopy (< 60 % cover)",
                    "detail": "Inspect these zones for missing plants, lodging, flooding or pest damage. Consider gap-filling "
                              "or replanting where establishment failed."})
    if weak.sum():
        rec.append({"priority": "medium", "title": f"{int(weak.sum())} zones with weak vigour despite full cover",
                    "detail": "The canopy is closed but less green or vigorous than the rest of the field. Possible causes: "
                              "nitrogen deficiency, water stress, disease or a different variety. Confirm with a leaf colour "
                              "chart/SPAD meter or soil test before applying fertiliser."})
    if s.get("ood_warning"):
        rec.append({"priority": "info", "title": "Image differs from training data",
                    "detail": "Treat the yield estimate as indicative. A multispectral (NIR) image or field samples improve reliability."})
    if not rec:
        rec.append({"priority": "info", "title": "Field looks uniform",
                    "detail": "No large problem areas were detected. Continue routine monitoring at the next growth stage."})
    rec.append({"priority": "info", "title": "Verify on the ground",
                "detail": "These are advisory prompts from imagery. Inspect the listed zones before taking management action."})

    order = "pred_t_ha" if zones.pred_t_ha.notna().any() else "rel_productivity"
    worst = zones[zones.poor_growth].sort_values([order, "veg_cover"]).head(10)
    scout = []
    tr = img.transform
    for z in worst.itertuples():
        cy, cx = z.zone_y + cell / 2, z.zone_x + cell / 2
        item = {"row": int(z.zone_y // cell) + 1, "col": int(z.zone_x // cell) + 1, "veg_cover_pct": 100 * z.veg_cover,
                "value": getattr(z, order), "value_name": "t/ha" if order == "pred_t_ha" else "relative productivity"}
        if tr and img.crs:
            # image pixel centre -> map coordinates (affine transform a,b,c,d,e,f); scale for any resizing
            sx = abs(tr[0]) and img.gsd_m / abs(tr[0]) if img.gsd_m else 1.0
            item["map_x"] = tr[2] + tr[0] * cx * sx
            item["map_y"] = tr[5] + tr[4] * cy * sx
            item["crs"] = img.crs
            try:
                from rasterio.warp import transform as _tf
                lon, lat = _tf(img.crs, "EPSG:4326", [item["map_x"]], [item["map_y"]])
                item["lat"], item["lon"] = round(lat[0], 7), round(lon[0], 7)
            except Exception:  # noqa: BLE001 - coordinates are a convenience
                pass
        scout.append(item)
    return rec, scout


def _json_safe(o):
    """Replace NaN/inf with None so the summary is valid JSON for browsers."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def _overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    green = np.zeros_like(rgb)
    green[..., 1] = 255
    out[mask] = (0.55 * rgb[mask] + 0.45 * green[mask]).astype(np.uint8)
    edges = cv2.Canny(mask.astype(np.uint8) * 255, 50, 150) > 0
    out[edges] = (255, 255, 0)
    return out


def _yield_map(rgb: np.ndarray, zones: pd.DataFrame, cell: int, rng, var: str = "pred_t_ha") -> np.ndarray:
    import matplotlib

    cmap = matplotlib.colormaps["RdYlGn"]
    out = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
    lo, hi = rng if var == "pred_t_ha" else (0.5, 1.5)
    for z in zones.itertuples():
        v = getattr(z, var)
        if not z.is_crop or not np.isfinite(v):
            continue
        c = np.array(cmap(np.clip((v - lo) / (hi - lo), 0, 1))[:3]) * 255
        sl = (slice(z.zone_y, z.zone_y + cell), slice(z.zone_x, z.zone_x + cell))
        out[sl] = (0.3 * out[sl] + 0.7 * c).astype(np.uint8)
    for z in zones.itertuples():           # grid lines (only when zones are large enough) + poor-growth outlines
        y, x = z.zone_y, z.zone_x
        if z.poor_growth:
            cv2.rectangle(out, (x, y), (x + cell - 1, y + cell - 1), (255, 0, 255), 2 if cell >= 48 else 1)
        elif cell >= 48:
            cv2.rectangle(out, (x, y), (x + cell - 1, y + cell - 1), (60, 60, 60), 1)
    return out


def save_products(res: AnalysisResult, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k, im in res.images.items():
        p = out_dir / f"{k}.png"
        cv2.imwrite(str(p), cv2.cvtColor(_thumb(im), cv2.COLOR_RGB2BGR))
        paths[k] = p.name
    res.zones.drop(columns=[c for c in res.zones.columns if c.startswith("ms_") or c.startswith("rgb_") or c.startswith("tex_")]) \
        .to_csv(out_dir / "zones.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(res.summary, indent=2, default=float))
    return paths


def _thumb(im: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = im.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else im
