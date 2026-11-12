"""Step 2b - check / estimate polygon-to-orthomosaic registration offsets.

The yield polygons are ~0.3-0.5 m^2, so a georeferencing offset of even one
metre between the polygons and an orthomosaic makes the pipeline sample the
neighbouring plot. For trials whose GeoJSON includes the dataset authors'
per-flight NDVI (IVIA2022, IVIA2023-F2), this script searches the (dx, dy)
shift that maximises the correlation between our plot-mean NDVI and theirs.

Method: rasterise the plot polygons into a label image on the orthomosaic grid
of the trial area, then for every candidate shift (coarse grid of +-6 m on a
4x down-sampled image, then +-0.25 m at full resolution) compute per-plot NDVI
means with np.bincount and their Pearson r with the reference NDVI.

Output: outputs/registration_offsets.json
    {raster_name: {"dx_m":..,"dy_m":..,"r_at_zero":..,"r_best":..,"applied":bool}}

Search range +-1.5 m (typical GNSS / orthomosaic georeferencing error).
An offset is APPLIED if it improves r by > 0.15. Otherwise zero is kept.
Rule history (kept for transparency): the first run searched +-6 m and also
required r > 0.6. The wide range aliased onto neighbouring plots of the
regular trial grid (2023-08-08 jumped by 6.8 m), so the range was narrowed to
the physically plausible +-1.5 m and the absolute-r condition was dropped.
No yield data is used at any point, so the correction cannot leak labels.
IVIA2023-F1 has no reference NDVI. It is in the same flights (same
orthomosaic) as F2, so it receives the same per-raster offset.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from rasterio.windows import from_bounds
from shapely.geometry import mapping, shape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smartyield import config  # noqa: E402
from smartyield.bestrice import load_plots  # noqa: E402

PAIRS = {"IVIA2022": "IVIA2022", "IVIA2023-F2": "IVIA2023"}
MARGIN_M = 3.0
SEARCH_M = 1.5


def flight_date(name: str) -> str:
    m = re.search(r"(20\d{2})-?(\d{2})-?(\d{2})", name)
    return f"{m[1]}-{m[2]}-{m[3]}"


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def plot_means(ndvi, labels, n, dy, dx):
    """Mean NDVI per label when the NDVI image is shifted by (dy, dx) pixels."""
    H, W = labels.shape
    ys, ye = max(0, -dy), min(H, H - dy)
    xs, xe = max(0, -dx), min(W, W - dx)
    lab = labels[ys:ye, xs:xe]
    val = ndvi[ys + dy:ye + dy, xs + dx:xe + dx]
    m = (lab > 0) & np.isfinite(val)
    s = np.bincount(lab[m], weights=val[m], minlength=n + 1)
    c = np.bincount(lab[m], minlength=n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s / c)[1:]


def main():
    plots, refs, _ = load_plots()
    out = {}
    for trial, cogdir in PAIRS.items():
        p = plots[plots.trial == trial].reset_index(drop=True)
        rf = refs[refs.trial == trial]
        for tif in sorted((config.COG_DIR / cogdir).glob("*.tif")):
            date = flight_date(tif.name)
            rd = rf[rf.date == date]
            if rd.empty:
                continue
            ref = p.merge(rd, on=["parcela", "n_dose"], how="left").ref_ndvi.to_numpy()
            with rasterio.open(tif) as ds:
                geoms = [shape(transform_geom(p.crs.iloc[0], ds.crs.to_string(), mapping(g)))
                         if p.crs.iloc[0] != ds.crs.to_string() else g for g in p.geometry]
                bx = np.array([g.bounds for g in geoms])
                win = from_bounds(bx[:, 0].min() - MARGIN_M, bx[:, 1].min() - MARGIN_M,
                                  bx[:, 2].max() + MARGIN_M, bx[:, 3].max() + MARGIN_M, ds.transform)
                win = win.round_offsets().round_lengths()
                red = ds.read(config.MAIA_BANDS["red"] + 1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
                nir = ds.read(config.MAIA_BANDS["nir"] + 1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
                tr = ds.window_transform(win)
                res = ds.res[0]
            with np.errstate(invalid="ignore", divide="ignore"):
                ndvi = (nir - red) / (nir + red)
            ndvi[~np.isfinite(ndvi) | ((nir + red) <= 0)] = np.nan
            labels = rasterize([(mapping(g), i + 1) for i, g in enumerate(geoms)], out_shape=ndvi.shape,
                               transform=tr, fill=0, dtype="int32")
            n = len(geoms)
            r0 = corr(plot_means(ndvi, labels, n, 0, 0), ref)
            # coarse search on 4x down-sampled grids (nearest for labels, mean for NDVI)
            f = 4
            H4, W4 = ndvi.shape[0] // f, ndvi.shape[1] // f
            with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore")
                nd4 = np.nanmean(ndvi[:H4 * f, :W4 * f].reshape(H4, f, W4, f), axis=(1, 3))
            lb4 = labels[f // 2:H4 * f:f, f // 2:W4 * f:f]
            R = int(SEARCH_M / (res * f))
            best = (r0, 0, 0)
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    r = corr(plot_means(nd4, lb4, n, dy, dx), ref)
                    if np.isfinite(r) and r > best[0]:
                        best = (r, dy * f, dx * f)
            # fine search at full resolution around the coarse optimum
            fine = best
            for dy in range(best[1] - f, best[1] + f + 1):
                for dx in range(best[2] - f, best[2] + f + 1):
                    r = corr(plot_means(ndvi, labels, n, dy, dx), ref)
                    if np.isfinite(r) and r > fine[0]:
                        fine = (r, dy, dx)
            r_best, dy, dx = fine
            # pixel shift (dy rows down, dx cols right) -> map offset of the image content
            dx_m, dy_m = dx * res, -dy * res
            applied = bool(r_best - r0 > 0.15)
            out[tif.name] = {"trial": trial, "date": date, "dx_m": round(dx_m, 3), "dy_m": round(dy_m, 3),
                             "r_at_zero": round(r0, 3), "r_best": round(r_best, 3), "applied": applied}
            print(f"{trial} {date}: r0={r0:.3f} best r={r_best:.3f} at dx={dx_m:+.2f} m dy={dy_m:+.2f} m "
                  f"-> {'APPLY' if applied else 'keep 0'}", flush=True)
    (config.OUTPUTS_DIR / "registration_offsets.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
