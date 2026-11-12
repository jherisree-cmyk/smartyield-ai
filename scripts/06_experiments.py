"""Step 6 - experiments 1, 3, 5, 6 (+ health calibration and demo data).

  exp1   preprocessing: visual comparison + measured noise/colour/contrast change,
         and the effect of preprocessing on ExG segmentation IoU (PhenoBench test)
  exp2b  U-Net vs NDVI-mask agreement on real rice MAIA imagery (no human
         masks exist for rice, so this is *agreement*, not accuracy)
  exp3   crop-condition analysis: vegetation features vs N dose and vs yield tercile
  exp5   generalisation: per-growth-stage accuracy, leave-site-season-out, temporal
  exp6   field-level: regional (block) and whole-trial yield from held-out predictions,
         plus a gridded yield map produced by the app pipeline on a real orthomosaic
  demo   writes data/demo/*.tif|png (real trial crops) used by exp6 and the app

All numbers written to outputs/experiments_summary.json are measured here.

Usage: python scripts/06_experiments.py all
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smartyield import config, indices  # noqa: E402
from smartyield.yield_model.core import metrics  # noqa: E402

FIG = config.FIGURES_DIR
OUT = config.OUTPUTS_DIR
DEMO = config.DATA_DIR / "demo"
SUMMARY = OUT / "experiments_summary.json"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.prop_cycle": matplotlib.cycler(color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"])})


def _save(key: str, val) -> None:
    s = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    s[key] = val
    SUMMARY.write_text(json.dumps(s, indent=2, default=float))


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer (1996) fast noise standard-deviation estimate."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = gray.shape
    conv = cv2.filter2D(gray.astype(np.float32), -1, k)
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2)))


# ------------------------------------------------------------------ demo data
def make_demo() -> None:
    """Crop the real IVIA-2022 trial (31 Aug flight) to a small GeoTIFF + an RGB PNG."""
    import rasterio
    from rasterio.windows import from_bounds

    from smartyield.bestrice import load_plots
    from smartyield.preprocessing import reflectance_to_rgb8

    DEMO.mkdir(parents=True, exist_ok=True)
    plots, _, _ = load_plots()
    jobs = [("IVIA2022", "20220831"), ("IRTA2022", "20220830_BESTRICE"), ("IVIA2023-F2", "2023-08-08")]
    made = []
    for trial, tag in jobs:
        d = config.COG_DIR / trial.split("-")[0]
        tifs = [t for t in d.glob("*.tif") if tag in t.name] if d.exists() else []
        if not tifs:
            continue
        p = plots[plots.trial == trial]
        bx = np.array([g.bounds for g in p.geometry])
        x0, y0, x1, y1 = bx[:, 0].min() - 1, bx[:, 1].min() - 1, bx[:, 2].max() + 1, bx[:, 3].max() + 1
        with rasterio.open(tifs[0]) as ds:
            if p.crs.iloc[0] != ds.crs.to_string():
                from rasterio.warp import transform_bounds
                x0, y0, x1, y1 = transform_bounds(p.crs.iloc[0], ds.crs, x0, y0, x1, y1)
            win = from_bounds(x0, y0, x1, y1, ds.transform).round_offsets().round_lengths()
            arr = ds.read(window=win).astype(np.float32)
            prof = ds.profile.copy()
            prof.update(width=win.width, height=win.height, transform=ds.window_transform(win), compress="deflate",
                        predictor=3, tiled=True, blockxsize=256, blockysize=256, driver="GTiff")
            prof.pop("blockysize", None) if win.height < 256 else None
        out = DEMO / f"{trial}_{tag.split('_')[0]}_trial_9band.tif"
        with rasterio.open(out, "w", **{k: v for k, v in prof.items() if k not in ("interleave",)}) as dst:
            dst.write(arr)
        rgb = reflectance_to_rgb8(np.moveaxis(arr, 0, -1))
        png = DEMO / f"{trial}_{tag.split('_')[0]}_trial_rgb.png"
        cv2.imwrite(str(png), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        made.append({"trial": trial, "tif": out.name, "png": png.name, "size_px": [int(win.height), int(win.width)],
                     "mb": round(out.stat().st_size / 1e6, 1)})
        print("demo:", made[-1])
    (DEMO / "README.md").write_text(
        "# Demo images (real data)\n\nCrops of the Best Rice MAIA orthomosaics (Zenodo 10.5281/zenodo.18768633, "
        "CC-BY-4.0) covering one yield trial each.\n\n* `*_9band.tif`: calibrated 9-band reflectance GeoTIFF "
        "(multispectral track, true NDVI)\n* `*_rgb.png`: RGB rendering of the same area (RGB track, no "
        "georeferencing: enter the field area manually)\n\n" + "\n".join(f"* {m}" for m in made) + "\n")


# ------------------------------------------------------------------ experiment 1
def exp1() -> None:
    from smartyield.preprocessing import preprocess_rgb
    from smartyield.segmentation.infer import exg_otsu_mask, seg_metrics

    test = sorted((config.PHENOBENCH_DIR / "val" / "images").glob("*.png"))
    samples = []
    if test:
        samples.append(("PhenoBench UAV (sugar beet)", cv2.cvtColor(cv2.imread(str(test[len(test) // 2])), cv2.COLOR_BGR2RGB)))
    for p in sorted(DEMO.glob("*_rgb.png"))[:1]:
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        samples.append(("MAIA rice trial (RGB render)", im[h // 3: h // 3 + 600, w // 4: w // 4 + 600]))
    # a real-world style colour cast (warm white balance) to show what gray-world removes
    if samples:
        cast = np.clip(samples[0][1].astype(np.float32) * [1.18, 1.0, 0.78], 0, 255).astype(np.uint8)
        samples.append(("same UAV image with warm colour cast", cast))
    rows = []
    fig, ax = plt.subplots(len(samples), 4, figsize=(13, 3.3 * len(samples)))
    ax = np.atleast_2d(ax)
    for r, (name, img) in enumerate(samples):
        pre = preprocess_rgb(img, color_correct=True)
        for c, k in enumerate(["original", "denoised", "analysis", "enhanced"]):
            ax[r, c].imshow(pre[k])
            ax[r, c].set_title(f"{name}\n{k}" if c == 0 else {"denoised": "bilateral denoise",
                                                               "analysis": "+ gray-world colour correction",
                                                               "enhanced": "+ CLAHE (display only)"}[k], fontsize=8)
            ax[r, c].axis("off")
        for k in ("original", "denoised", "analysis", "enhanced"):
            g = cv2.cvtColor(pre[k], cv2.COLOR_RGB2GRAY)
            m = pre[k].reshape(-1, 3).mean(0)
            rows.append({"image": name, "stage": k, "noise_sigma": noise_sigma(g), "rms_contrast": float(g.std()),
                         "channel_mean_spread": float(m.max() - m.min())})
    plt.tight_layout()
    plt.savefig(FIG / "exp1_preprocessing.png", dpi=110)
    plt.close()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp1_preprocessing_stats.csv", index=False)

    # effect on ExG+Otsu segmentation (labelled PhenoBench test images), with and without a colour cast
    res = {}
    names = test[:120]
    for cond in ("as captured", "warm colour cast"):
        accs = {"raw": np.zeros(4), "preprocessed": np.zeros(4)}
        for p in names:
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            if cond == "warm colour cast":
                img = np.clip(img.astype(np.float32) * [1.18, 1.0, 0.78], 0, 255).astype(np.uint8)
            sem = cv2.imread(str(p).replace("images", "semantics"), cv2.IMREAD_UNCHANGED)
            gt = cv2.resize(sem, (512, 512), interpolation=cv2.INTER_NEAREST) > 0
            for k, im in (("raw", img), ("preprocessed", preprocess_rgb(img, color_correct=True)["analysis"])):
                m = seg_metrics(exg_otsu_mask(im), gt)
                accs[k] += [m["tp"], m["fp"], m["fn"], m["tn"]]
        res[cond] = {k: {"iou": v[0] / (v[0] + v[1] + v[2]), "dice": 2 * v[0] / (2 * v[0] + v[1] + v[2])}
                     for k, v in accs.items()}
    out = {"stats": df.round(3).to_dict(orient="records"), "exg_segmentation_effect": res, "n_images": len(names)}
    _save("exp1_preprocessing", out)
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------ experiment 2b
def exp2b() -> None:
    from smartyield.segmentation.infer import UNET_PATH, exg_otsu_mask, ndvi_otsu_mask, seg_metrics, unet_probability

    if not UNET_PATH.exists():
        print("no U-Net yet")
        return
    import rasterio
    rng = np.random.default_rng(0)
    rows = []
    for tif in sorted(DEMO.glob("*_9band.tif")):
        with rasterio.open(tif) as ds:
            ms = np.moveaxis(ds.read().astype(np.float32), 0, -1)
        ok = np.isfinite(ms).all(-1) & (ms.sum(-1) > 0)
        ms[~ok] = 0
        from smartyield.preprocessing import reflectance_to_rgb8
        rgb = reflectance_to_rgb8(ms)
        ref = ndvi_otsu_mask(ms) & ok
        un = (unet_probability(rgb) > 0.5) & ok
        ex = exg_otsu_mask(rgb) & ok
        rows.append({"image": tif.name, "ndvi_cover": float(ref[ok].mean()),
                     "unet_vs_ndvi": seg_metrics(un[ok], ref[ok]), "exg_vs_ndvi": seg_metrics(ex[ok], ref[ok])})
        h, w = rgb.shape[:2]
        y, x = h // 2 - 200, w // 2 - 200
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
        for a, (im, t) in zip(ax, [(rgb[y:y + 400, x:x + 400], "MAIA RGB render"),
                                   (ref[y:y + 400, x:x + 400], "NDVI-Otsu (reference)"),
                                   (ex[y:y + 400, x:x + 400], f"ExG-Otsu  agreement IoU {rows[-1]['exg_vs_ndvi']['iou']:.2f}"),
                                   (un[y:y + 400, x:x + 400], f"U-Net  agreement IoU {rows[-1]['unet_vs_ndvi']['iou']:.2f}")]):
            a.imshow(im, cmap="gray")
            a.set_title(t, fontsize=8)
            a.axis("off")
        plt.tight_layout()
        plt.savefig(FIG / f"exp2b_rice_agreement_{tif.stem}.png", dpi=110)
        plt.close()
    _save("exp2b_rice_mask_agreement", rows)
    for r in rows:
        print(r["image"], "U-Net IoU vs NDVI %.3f" % r["unet_vs_ndvi"]["iou"], "ExG %.3f" % r["exg_vs_ndvi"]["iou"])


# ------------------------------------------------------------------ health calibration + experiment 3
def exp3() -> None:
    df = pd.read_csv(config.PROCESSED_DIR / "plot_features_ms.csv", low_memory=False)
    # data-driven NDVI class breaks (see docs: absolute NDVI of this calibrated product is ~0.25 lower
    # than the dataset authors' published NDVI, so textbook breaks do not transfer)
    nd = df.ms_ndvi_veg_mean.dropna()
    breaks = {"poor_below": float(nd.quantile(0.20)), "healthy_from": float(nd.quantile(0.60)),
              "background_below": 0.30, "source": "20th/60th percentiles of plot NDVI (vegetation pixels), "
              "Best Rice training imagery, all flights"}
    (config.MODELS_DIR / "health_calibration.json").write_text(json.dumps(breaks, indent=2))

    feats = ["ms_ndvi_veg_mean", "ms_ndre_veg_mean", "ms_cire_mean", "veg_cover", "rgb_exg_mean", "rgb_vari_mean",
             "tex_contrast"]
    # one flight per site-season: the one with the highest NDVI-yield correlation is NOT used (selection bias);
    # we use the flight closest to mid-season (median DOY of that site-season's flights)
    rows, tests = [], []
    for ss, g in df.groupby("site_year"):
        mid = sorted(g.flight_date.unique())[len(g.flight_date.unique()) // 2]
        gm = g[g.flight_date == mid].copy()
        gm["tercile"] = pd.qcut(gm.yield_t_ha, 3, labels=["low", "mid", "high"])
        for f in feats:
            a, b = gm[gm.n_dose == 100][f].dropna(), gm[gm.n_dose == 200][f].dropna()
            u = stats.mannwhitneyu(a, b) if len(a) > 5 and len(b) > 5 else None
            lo, hi = gm[gm.tercile == "low"][f].dropna(), gm[gm.tercile == "high"][f].dropna()
            kw = stats.kruskal(*[gm[gm.tercile == t][f].dropna() for t in ("low", "mid", "high")])
            rho = stats.spearmanr(gm[f], gm.yield_t_ha, nan_policy="omit")
            tests.append({"site_season": ss, "flight": mid, "feature": f,
                          "mean_N100": float(a.mean()), "mean_N200": float(b.mean()),
                          "mannwhitney_p_N": float(u.pvalue) if u else None,
                          "mean_low_yield": float(lo.mean()), "mean_high_yield": float(hi.mean()),
                          "kruskal_p_tercile": float(kw.pvalue), "spearman_rho_yield": float(rho.statistic)})
        gm["health_class"] = np.where(gm.ms_ndvi_veg_mean < breaks["poor_below"], "poor",
                                      np.where(gm.ms_ndvi_veg_mean >= breaks["healthy_from"], "healthy", "moderate"))
        q20, q60 = gm.ms_ndvi_veg_mean.quantile([0.2, 0.6])
        gm["relative_class"] = np.where(gm.ms_ndvi_veg_mean < q20, "poor",
                                        np.where(gm.ms_ndvi_veg_mean >= q60, "healthy", "moderate"))
        for scheme, col in (("absolute (pooled-calibrated)", "health_class"), ("relative within image (app)", "relative_class")):
            for c, gg in gm.groupby(col):
                rows.append({"scheme": scheme, "site_season": ss, "flight": mid, "health_class": c, "n_plots": len(gg),
                             "mean_yield_t_ha": float(gg.yield_t_ha.mean()), "sd_yield": float(gg.yield_t_ha.std())})
    t = pd.DataFrame(tests)
    t.to_csv(OUT / "exp3_condition_tests.csv", index=False)
    hc = pd.DataFrame(rows)
    hc.to_csv(OUT / "exp3_health_class_yield.csv", index=False)

    ss_list = sorted(df.site_year.unique())
    fig, ax = plt.subplots(2, len(ss_list), figsize=(4.2 * len(ss_list), 6.4), squeeze=False)
    for i, ss in enumerate(ss_list):
        g = df[df.site_year == ss]
        mid = sorted(g.flight_date.unique())[len(g.flight_date.unique()) // 2]
        gm = g[g.flight_date == mid].copy()
        gm["tercile"] = pd.qcut(gm.yield_t_ha, 3, labels=["low", "mid", "high"])
        data = [gm[gm.tercile == k].ms_ndvi_veg_mean.dropna() for k in ("low", "mid", "high")]
        ax[0, i].boxplot(data, tick_labels=["low", "mid", "high"])
        ax[0, i].set_title(f"{ss} ({mid}): NDVI by yield tercile")
        ax[0, i].set_ylabel("plot NDVI (vegetation)")
        data = [gm[gm.n_dose == n].ms_ndvi_veg_mean.dropna() for n in (100, 200)]
        ax[1, i].boxplot(data, tick_labels=["N 100", "N 200"])
        ax[1, i].set_title(f"{ss}: NDVI by nitrogen dose")
    plt.tight_layout()
    plt.savefig(FIG / "exp3_crop_condition.png", dpi=110)
    plt.close()
    _save("exp3_crop_condition", {"health_calibration": breaks, "tests": t.round(4).to_dict(orient="records"),
                                  "health_class_yield": hc.round(3).to_dict(orient="records")})
    print(t[["site_season", "feature", "mannwhitney_p_N", "kruskal_p_tercile", "spearman_rho_yield"]].round(4).to_string())
    print(hc.round(2).to_string())


# ------------------------------------------------------------------ experiment 5
def exp5() -> None:
    out = {}
    for track in ("ms", "rgb"):
        p = OUT / f"yield_predictions_{track}.csv"
        rj = OUT / f"yield_results_{track}.json"
        if not p.exists():
            continue
        oof = pd.read_csv(p)
        res = json.loads(rj.read_text())
        best = res["selected_model"]
        per_date = []
        for (ss, d), g in oof.groupby(["site_year", "flight_date"]):
            m = metrics(g.yield_t_ha, g[f"P1_{best}"])
            per_date.append({"site_season": ss, "flight_date": d, "doy": int(g.doy.iloc[0]), **m})
        out[track] = {"selected_model": best, "per_flight_P1": per_date,
                      "P2": {s: r["image_level"] for s, r in res["P2_leave_site_season_out"].get(best, {}).items()},
                      "P3": res["P3_temporal_2022_to_2023"].get(best)}
    if not out:
        return
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for track, mk in (("ms", "o-"), ("rgb", "s--")):
        if track not in out:
            continue
        d = pd.DataFrame(out[track]["per_flight_P1"])
        for ss, g in d.groupby("site_season"):
            ax.plot(g.doy, g.R2, mk, label=f"{ss} – {'multispectral' if track == 'ms' else 'RGB'}")
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xlabel("flight day of year (crop development →)")
    ax.set_ylabel("R² (grouped CV, image level)")
    ax.set_title("Yield predictability by flight date (growth stage)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG / "exp5_r2_by_growth_stage.png", dpi=110)
    plt.close()
    _save("exp5_generalisation", out)
    print(json.dumps({k: {"P2": v["P2"], "P3": v["P3"]} for k, v in out.items()}, indent=1, default=float)[:3000])


# ------------------------------------------------------------------ experiment 4 figures
def exp4_figures() -> None:
    names = ["Mean baseline", "Linear Regression", "Ridge Regression", "Random Forest", "SVR", "XGBoost"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    w = 0.38
    for i, track in enumerate(("ms", "rgb")):
        rj = OUT / f"yield_results_{track}.json"
        if not rj.exists():
            continue
        r = json.loads(rj.read_text())["P1_grouped_cv"]
        x = np.arange(len(names)) + (i - 0.5) * w
        ax[0].bar(x, [r[n]["image_level"]["RMSE"] for n in names], w, label="multispectral" if track == "ms" else "RGB only")
        ax[1].bar(x, [r[n]["image_level"]["R2"] for n in names], w, label="multispectral" if track == "ms" else "RGB only")
    for a, t in zip(ax, ("RMSE (t/ha) – lower is better", "R² – higher is better")):
        a.set_xticks(range(len(names)), names, rotation=15)
        a.set_title(t)
        a.legend()
    ax[1].axhline(0, color="#898781", lw=1)
    plt.suptitle("Experiment 4 – yield models, grouped 5-fold CV (blocks held out)")
    plt.tight_layout()
    plt.savefig(FIG / "exp4_model_comparison.png", dpi=110)
    plt.close()
    for track in ("ms", "rgb"):
        p = OUT / f"yield_predictions_{track}.csv"
        if not p.exists():
            continue
        oof = pd.read_csv(p)
        res = json.loads((OUT / f"yield_results_{track}.json").read_text())
        best = res["selected_model"]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        for a, lvl in zip(ax, ("image", "plot")):
            if lvl == "image":
                yt, yp, ss = oof.yield_t_ha, oof[f"P1_{best}"], oof.site_year
            else:
                g = oof.groupby("plot_uid").agg(y=("yield_t_ha", "first"), p=(f"P1_{best}", "mean"), s=("site_year", "first"))
                yt, yp, ss = g.y, g.p, g.s
            for s in sorted(ss.unique()):
                k = ss == s
                a.scatter(yt[k], yp[k], s=8, alpha=0.5, label=s)
            lim = [0, max(yt.max(), yp.max()) + 0.5]
            a.plot(lim, lim, color="#52514e", lw=1)
            m = metrics(yt, yp)
            a.set_title(f"{lvl} level: RMSE {m['RMSE']:.2f} t/ha, R² {m['R2']:.2f}")
            a.set_xlabel("actual yield (t/ha)")
            a.set_ylabel("predicted yield (t/ha)")
            a.legend(fontsize=7)
        plt.suptitle(f"{best} ({'multispectral' if track == 'ms' else 'RGB only'}) – out-of-fold predictions")
        plt.tight_layout()
        plt.savefig(FIG / f"exp4_pred_vs_actual_{track}.png", dpi=110)
        plt.close()
        imp = pd.Series(res["permutation_importance"]).head(15)[::-1]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.barh(imp.index, imp.values)
        ax.set_xlabel("RMSE increase when feature is shuffled (t/ha)")
        ax.set_title(f"Permutation importance – {best} ({track})")
        plt.tight_layout()
        plt.savefig(FIG / f"exp4_feature_importance_{track}.png", dpi=110)
        plt.close()


# ------------------------------------------------------------------ experiment 6
def exp6() -> None:
    out = {}
    for track in ("ms", "rgb"):
        p = OUT / f"yield_predictions_{track}.csv"
        if not p.exists():
            continue
        oof = pd.read_csv(p)
        best = json.loads((OUT / f"yield_results_{track}.json").read_text())["selected_model"]
        col = f"P2_{best}" if f"P2_{best}" in oof else f"P1_{best}"
        plot = oof.groupby("plot_uid").agg(site_year=("site_year", "first"), block=("block", "first"),
                                           n_dose=("n_dose", "first"), y=("yield_t_ha", "first"), p=(col, "mean"))
        reg = plot.groupby(["site_year", "block", "n_dose"]).agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size")).reset_index()
        field = plot.groupby("site_year").agg(actual_mean=("y", "mean"), pred_mean=("p", "mean"), n_plots=("y", "size"))
        field["abs_error_t_ha"] = (field.pred_mean - field.actual_mean).abs()
        field["pct_error"] = 100 * field.abs_error_t_ha / field.actual_mean
        out[track] = {"protocol": col, "plot_level": metrics(plot.y, plot.p),
                      "regional_block_x_N_level": metrics(reg.y, reg.p), "n_regions": len(reg),
                      "regional_spearman": float(stats.spearmanr(reg.y, reg.p).statistic),
                      "field_level": field.round(3).reset_index().to_dict(orient="records")}
    # gridded yield map on a real orthomosaic crop with the app pipeline
    from smartyield.analysis import analyze, save_products
    from smartyield.bestrice import load_plots

    plots, _, _ = load_plots()
    maps = []
    for tif in sorted(DEMO.glob("*_9band.tif")):
        trial = tif.name.split("_")[0]
        flight = tif.name.split("_")[1]
        from datetime import date
        fd = date(int(flight[:4]), int(flight[4:6]), int(flight[6:8])) if flight.isdigit() else date.fromisoformat(flight)
        res = analyze(tif, capture_date=fd)
        save_products(res, OUT / "exp6_maps" / tif.stem)
        gm = plots[plots.trial == trial]
        maps.append({"image": tif.name, "zones": res.summary["n_zones"], "crop_zones": res.summary["n_crop_zones"],
                     "zone_size_m": res.summary["zone_size_m"], "pred_field_t_ha": res.summary["predicted_yield_t_ha"],
                     "pi90": [res.summary["yield_lo90_t_ha"], res.summary["yield_hi90_t_ha"]],
                     "actual_trial_mean_t_ha": float(gm.yield_t_ha.mean()), "cropped_area_ha": res.summary["cropped_area_ha"],
                     "production_t": res.summary["estimated_production_t"],
                     "note": "Model trained on all site-seasons incl. this one (map demo); held-out accuracy is in the "
                             "field_level table above."})
        print("map", maps[-1])
        im = res.images
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        for a, (k, t) in zip(ax, (("original", "orthomosaic (RGB render)"), ("health", "health map"),
                                  ("yield_map", "zone yield map (1 m zones)"))):
            a.imshow(im[k])
            a.set_title(t)
            a.axis("off")
        plt.suptitle(f"{tif.stem}: predicted {res.summary['predicted_yield_t_ha']:.2f} t/ha vs measured trial mean "
                     f"{gm.yield_t_ha.mean():.2f} t/ha")
        plt.tight_layout()
        plt.savefig(FIG / f"exp6_field_map_{trial}.png", dpi=100)
        plt.close()
    out["gridded_maps"] = maps
    _save("exp6_field_level", out)
    print(json.dumps({k: v for k, v in out.items() if k != "gridded_maps"}, indent=1, default=float)[:2500])


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = {"demo": make_demo, "exp1": exp1, "exp2b": exp2b, "exp3": exp3, "exp4": exp4_figures, "exp5": exp5,
             "exp6": exp6}
    for k, fn in steps.items():
        if what in (k, "all"):
            print(f"===== {k}")
            fn()
