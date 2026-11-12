"""Step 2 - build the structured feature dataset (one row per plot x flight date).

For every orthomosaic (one drone flight) and every yield plot polygon:
  1. reproject the polygon to the raster CRS (if needed) and apply the
     registration offset estimated by scripts/02b_estimate_registration.py
  2. read a window around the plot (all 9 MAIA bands)
  3. rasterise the polygon -> region mask
  4. vegetation mask
       ms track : NDVI > Otsu threshold computed on the whole orthomosaic
       rgb track: the RGB segmentation method the app uses (--rgb-seg)
  5. smartyield.features.region_features(...) -> feature vector
  6. attach the measured yield (t/ha) of that plot

Outputs
  data/processed/plot_features_ms.csv   (multispectral track)
  data/processed/plot_features_rgb.csv  (RGB-only track, NIR never touched)
  outputs/ndvi_validation.json          (our NDVI vs dataset-provided NDVI)
  outputs/data_cleaning_log.json

Leakage note: a row is (plot, date). All rows of one plot share one yield, so
splits must be grouped by plot / block / site / season. They are never random
row splits (see scripts/05_train_yield.py).

Usage: python scripts/02_build_plot_dataset.py --rgb-seg unet --workers 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smartyield import config  # noqa: E402
from smartyield.bestrice import load_plots  # noqa: E402

TRIAL_TO_COGDIR = {"IVIA2022": "IVIA2022", "IRTA2022": "IRTA2022", "IVIA2023-F1": "IVIA2023",
                   "IVIA2023-F2": "IVIA2023", "IRTA2023": "IRTA2023"}
CONTEXT_M = 0.6   # metres of context read around each plot (for texture + U-Net receptive field)


def flight_date(name: str) -> date:
    m = re.search(r"(20\d{2})-?(\d{2})-?(\d{2})", name)
    if not m:
        raise ValueError(f"no date in {name}")
    return date(int(m[1]), int(m[2]), int(m[3]))


def global_thresholds(path: Path) -> dict:
    """NDVI and ExG Otsu thresholds from a decimated overview of the whole orthomosaic."""
    import rasterio
    from rasterio.enums import Resampling

    from smartyield import indices
    from smartyield.preprocessing import reflectance_to_rgb8
    from smartyield.segmentation.infer import otsu_threshold

    with rasterio.open(path) as ds:
        f = max(1, max(ds.width, ds.height) // 2048)
        arr = ds.read(out_shape=(ds.count, ds.height // f, ds.width // f), resampling=Resampling.average)
    ms = np.moveaxis(arr, 0, -1).astype(np.float32)[..., :9]
    ok = np.isfinite(ms).all(-1) & (ms.sum(-1) > 0)
    nd = indices.multispectral_indices(ms, config.MAIA_BANDS)["ndvi"]
    nd[~ok] = np.nan
    exg = indices.rgb_indices(reflectance_to_rgb8(ms))["exg"]
    exg[~ok] = np.nan
    return {"ndvi_t": max(otsu_threshold(nd), 0.3), "exg_t": otsu_threshold(exg),
            "ndvi_overview_median": float(np.nanmedian(nd))}


def process_raster(path: str, plots: pd.DataFrame, rgb_seg: str) -> tuple[list, list, dict]:
    import cv2
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.warp import transform_geom
    from rasterio.windows import from_bounds
    from shapely.geometry import mapping, shape

    from smartyield import indices
    from smartyield.features import region_features
    from smartyield.preprocessing import denoise, reflectance_to_rgb8
    from smartyield.segmentation.infer import unet_probability

    from shapely.affinity import translate

    path = Path(path)
    d = flight_date(path.name)
    reg_p = config.OUTPUTS_DIR / "registration_offsets.json"
    reg = json.loads(reg_p.read_text()).get(path.name, {}) if reg_p.exists() else {}
    shift = (reg["dx_m"], reg["dy_m"]) if reg.get("applied") else (0.0, 0.0)
    doy = d.timetuple().tm_yday
    th = global_thresholds(path)
    ms_rows, rgb_rows = [], []
    with rasterio.open(path) as ds:
        rcrs = ds.crs.to_string()
        b = ds.bounds
        for r in plots.itertuples():
            geom = r.geometry
            if r.crs and r.crs != rcrs:
                geom = shape(transform_geom(r.crs, rcrs, mapping(geom)))
            if shift != (0.0, 0.0):          # registration correction (scripts/02b)
                geom = translate(geom, xoff=shift[0], yoff=shift[1])
            x0, y0, x1, y1 = geom.bounds
            if x0 < b.left or x1 > b.right or y0 < b.bottom or y1 > b.top:
                continue
            win = from_bounds(x0 - CONTEXT_M, y0 - CONTEXT_M, x1 + CONTEXT_M, y1 + CONTEXT_M, ds.transform)
            win = win.round_offsets().round_lengths()
            arr = ds.read(window=win, boundless=True, fill_value=np.nan).astype(np.float32)
            ms = np.moveaxis(arr, 0, -1)[..., :9]
            if ms.shape[0] < 8 or ms.shape[1] < 8:
                continue
            region = geometry_mask([mapping(geom)], out_shape=ms.shape[:2],
                                   transform=ds.window_transform(win), invert=True)
            ok = np.isfinite(ms).all(-1) & (ms.sum(-1) > 0)
            if (region & ok).sum() < 0.8 * region.sum() or region.sum() < 20:
                continue                                    # plot not (fully) imaged on this flight
            ms[~ok] = 0
            rgb_raw = reflectance_to_rgb8(ms)
            rgb = denoise(rgb_raw)          # same "analysis" image the app derives from an RGB upload
            nd = indices.multispectral_indices(ms, config.MAIA_BANDS)["ndvi"]
            veg_ms = np.nan_to_num(nd, nan=-9) > th["ndvi_t"]
            if rgb_seg == "unet":
                veg_rgb = unet_probability(rgb, tile=max(64, ((max(rgb.shape[:2]) + 31) // 32) * 32)) > 0.5
            else:
                veg_rgb = np.nan_to_num(indices.rgb_indices(rgb)["exg"], nan=-9) > th["exg_t"]
            veg_rgb = cv2.morphologyEx(veg_rgb.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
            meta = {"plot_uid": r.plot_uid, "trial": r.trial, "site": r.site, "year": r.year,
                    "site_year": r.site_year, "parcela": r.parcela, "block": r.block, "n_dose": r.n_dose,
                    "variety": r.variety, "flight_date": d.isoformat(), "raster": path.name,
                    "yield_t_ha": r.yield_t_ha, "plot_area_m2": geom.area}
            ms_rows.append({**meta, **region_features(rgb_raw, veg_ms, region, ms=ms, doy=doy)})
            rgb_rows.append({**meta, **region_features(rgb, veg_rgb, region, ms=None, doy=doy)})
    return ms_rows, rgb_rows, {"raster": path.name, "date": d.isoformat(), **th, "plots": len(ms_rows),
                               "registration_shift_m": list(shift)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-seg", choices=["unet", "exg_otsu"], default=config.RGB_SEG_METHOD)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--trials", nargs="+", default=None)
    ap.add_argument("--append", action="store_true",
                    help="keep existing rows of other trials in the output CSVs (incremental builds)")
    a = ap.parse_args()

    plots, refs, logs = load_plots()
    (config.OUTPUTS_DIR / "data_cleaning_log.json").write_text(json.dumps([vars(x) for x in logs], indent=2))
    if a.trials:
        plots = plots[plots.trial.isin(a.trials)]

    jobs = []
    for cogdir, grp in plots.groupby(plots.trial.map(TRIAL_TO_COGDIR)):
        d = config.COG_DIR / cogdir
        tifs = sorted(d.glob("*.tif")) if d.exists() else []
        if not tifs:
            print(f"[skip] no imagery for {cogdir} ({len(grp)} plots)")
            continue
        # IRTA 2022 has two mosaics per date (BESTRICE trial and SUELOS soil trial); the yield
        # polygons belong to the BESTRICE trial, the bounds check discards the other automatically.
        for t in tifs:
            jobs.append((str(t), grp))
    print(f"{len(jobs)} raster jobs, {len(plots)} plots")

    ms_all, rgb_all, rinfo = [], [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process_raster, p, g, a.rgb_seg): p for p, g in jobs}
        for fu in as_completed(futs):
            ms_rows, rgb_rows, info = fu.result()
            ms_all += ms_rows
            rgb_all += rgb_rows
            rinfo.append(info)
            print(f"  {info['raster']}: {info['plots']} plots  NDVI_t={info['ndvi_t']:.3f}", flush=True)

    ms_df = pd.DataFrame(ms_all)
    rgb_df = pd.DataFrame(rgb_all)
    if a.append:
        for name, new in (("ms", ms_df), ("rgb", rgb_df)):
            old_p = config.PROCESSED_DIR / f"plot_features_{name}.csv"
            if old_p.exists():
                old = pd.read_csv(old_p)
                old = old[~old.trial.isin(new.trial.unique())]
                merged = pd.concat([old, new], ignore_index=True)
                if name == "ms":
                    ms_df = merged
                else:
                    rgb_df = merged
        inv_p = config.OUTPUTS_DIR / "raster_inventory.json"
        if inv_p.exists():
            names = {r["raster"] for r in rinfo}
            rinfo += [r for r in json.loads(inv_p.read_text()) if r["raster"] not in names]
    ms_df = ms_df.sort_values(["trial", "plot_uid", "flight_date"])
    rgb_df = rgb_df.sort_values(["trial", "plot_uid", "flight_date"])
    ms_df.to_csv(config.PROCESSED_DIR / "plot_features_ms.csv", index=False)
    rgb_df.to_csv(config.PROCESSED_DIR / "plot_features_rgb.csv", index=False)
    (config.OUTPUTS_DIR / "raster_inventory.json").write_text(json.dumps(sorted(rinfo, key=lambda x: x["raster"]), indent=2))
    print("rows:", len(ms_df), "plots:", ms_df.plot_uid.nunique(), "by trial/date:")
    print(ms_df.groupby(["trial", "flight_date"]).size())

    # ---- validate band order + georeferencing against dataset-provided NDVI
    if len(refs):
        m = ms_df.merge(refs.rename(columns={"date": "flight_date"}), on=["trial", "parcela", "n_dose", "flight_date"])
        if len(m):
            r = float(np.corrcoef(m.ms_ndvi_mean, m.ref_ndvi)[0, 1])
            val = {"n": len(m), "pearson_r": r, "mae": float(np.mean(np.abs(m.ms_ndvi_mean - m.ref_ndvi))),
                   "bias": float(np.mean(m.ms_ndvi_mean - m.ref_ndvi)),
                   "per_trial": m.groupby("trial").apply(
                       lambda g: {"n": len(g), "r": float(np.corrcoef(g.ms_ndvi_mean, g.ref_ndvi)[0, 1])}).to_dict()}
            (config.OUTPUTS_DIR / "ndvi_validation.json").write_text(json.dumps(val, indent=2))
            print("NDVI validation vs dataset-provided values:", json.dumps(val, indent=1))


if __name__ == "__main__":
    main()
