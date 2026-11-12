"""Step 5 - train, tune, compare and save the yield regression models
(Experiments 4 and 5).

Validation protocols (all grouped, so no plot ever appears in both train and test):
  P1  Grouped 5-fold CV, groups = trial x replicate block (spatially separate
      blocks). Used to SELECT the best model.
  P2  Leave-one-site-season-out (LOSO): train on all other site-seasons, test
      on an unseen one (IVIA2022 / IRTA2022 / IVIA2023). Generalisation to new
      fields and new seasons.
  P3  Temporal: train on the 2022 season, test on the 2023 season.

Hyper-parameters are tuned with an inner GroupKFold(3) grid search on the
training part of every outer split (nested CV). The test fold is never seen
during tuning.

Metrics are reported at two levels:
  * image level: one prediction per (plot, flight). This is what the app does
    for a single upload.
  * plot level : the mean of a plot's predictions over its flights
    (multi-temporal monitoring).

Usage: python scripts/05_train_yield.py --track ms
       python scripts/05_train_yield.py --track rgb
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smartyield import config  # noqa: E402
from smartyield.features import feature_columns  # noqa: E402
from smartyield.yield_model.core import MODEL_NAMES, YieldBundle, add_relative_features, make_model, metrics  # noqa: E402

warnings.filterwarnings("ignore")


def season_of(df: pd.DataFrame) -> pd.Series:
    return df.site_year  # IVIA2023-F1 and -F2 share one flight campaign -> same season group


def fit_tuned(name, X, y, groups):
    pipe, grid = make_model(name)
    if not grid:
        return clone(pipe).fit(X, y), {}
    n_groups = len(np.unique(groups))
    gs = GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=min(3, n_groups)), scoring="neg_root_mean_squared_error",
                      n_jobs=1 if name in ("Random Forest", "XGBoost") else -1, refit=True)
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_, gs.best_params_


def run_split(name, df, X, y, tr, te, inner_groups):
    model, params = fit_tuned(name, X.iloc[tr], y.iloc[tr], inner_groups.iloc[tr])
    pred = model.predict(X.iloc[te])
    return model, params, pred


def plot_level(df_te: pd.DataFrame, pred: np.ndarray) -> dict:
    t = df_te[["plot_uid", "yield_t_ha"]].copy()
    t["pred"] = pred
    g = t.groupby("plot_uid").mean()
    return metrics(g.yield_t_ha, g.pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["ms", "rgb"], required=True)
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES)
    ap.add_argument("--feature-mode", choices=["abs", "absrel"], default="abs",
                    help="abs: absolute features; absrel: + image-relative features (value - image median)")
    ap.add_argument("--tag", default="", help="suffix for output files (ablation runs)")
    a = ap.parse_args()
    t0 = time.time()

    df = pd.read_csv(config.PROCESSED_DIR / f"plot_features_{a.track}.csv")
    df = df[df.n_pixels >= 20].reset_index(drop=True)
    feats = feature_columns(a.track, df.columns)
    feats = [c for c in feats if df[c].notna().mean() > 0.9 and df[c].std() > 0]
    if a.feature_mode == "absrel":
        # one drone image = one (site-season, flight date); IVIA2023 F1/F2 share a flight
        df = add_relative_features(df, feats, df.site_year + "_" + df.flight_date)
        feats = feats + [f"{c}_rel" for c in feats if c != "doy"]
    X, y = df[feats].astype(float), df.yield_t_ha.astype(float)
    block_groups = df.trial + "_B" + df.block.astype(int).astype(str)
    seasons = season_of(df)
    print(f"track={a.track} rows={len(df)} plots={df.plot_uid.nunique()} features={len(feats)} "
          f"site-seasons={sorted(seasons.unique())}")

    results = {"track": a.track, "feature_mode": a.feature_mode, "n_rows": len(df), "n_plots": int(df.plot_uid.nunique()),
               "features": feats, "site_seasons": sorted(seasons.unique()), "P1_grouped_cv": {},
               "P2_leave_site_season_out": {}, "P3_temporal_2022_to_2023": {}, "best_params": {}}
    oof = pd.DataFrame({"plot_uid": df.plot_uid, "trial": df.trial, "site_year": df.site_year,
                        "flight_date": df.flight_date, "doy": df.get("doy"), "block": df.block,
                        "n_dose": df.n_dose, "yield_t_ha": y})

    # ---------------- P1 grouped CV (model selection)
    n_folds = min(5, block_groups.nunique())
    gkf = GroupKFold(n_splits=n_folds)
    for name in a.models:
        pred = np.zeros(len(df))
        params_all = []
        for k, (tr, te) in enumerate(gkf.split(X, y, block_groups)):
            _, params, p = run_split(name, df, X, y, tr, te, block_groups)
            pred[te] = p
            params_all.append(params)
        oof[f"P1_{name}"] = pred
        results["P1_grouped_cv"][name] = {"image_level": metrics(y, pred), "plot_level": plot_level(df, pred)}
        results["best_params"][name] = params_all
        m = results["P1_grouped_cv"][name]["image_level"]
        print(f"[P1] {name:18s} RMSE {m['RMSE']:.3f} MAE {m['MAE']:.3f} R2 {m['R2']:.3f} MAPE {m['MAPE_%']:.1f}% "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---------------- P2 leave-one-site-season-out
    if seasons.nunique() > 1:
        for name in a.models:
            res, pred_all = {}, np.full(len(df), np.nan)
            for s in sorted(seasons.unique()):
                te = np.where(seasons == s)[0]
                tr = np.where(seasons != s)[0]
                _, _, p = run_split(name, df, X, y, tr, te, block_groups)
                pred_all[te] = p
                res[s] = {"image_level": metrics(y.iloc[te], p), "plot_level": plot_level(df.iloc[te], p),
                          "actual_mean_t_ha": float(y.iloc[te].mean()), "pred_mean_t_ha": float(np.mean(p))}
            oof[f"P2_{name}"] = pred_all
            results["P2_leave_site_season_out"][name] = res
            print(f"[P2] {name:18s} " + "  ".join(f"{s}: R2 {r['image_level']['R2']:.2f} RMSE {r['image_level']['RMSE']:.2f}"
                                                  for s, r in res.items()), flush=True)

    # ---------------- P3 temporal 2022 -> 2023
    if set(df.year.unique()) >= {2022, 2023}:
        tr = np.where(df.year == 2022)[0]
        te = np.where(df.year == 2023)[0]
        for name in a.models:
            _, _, p = run_split(name, df, X, y, tr, te, block_groups)
            results["P3_temporal_2022_to_2023"][name] = {"image_level": metrics(y.iloc[te], p),
                                                         "plot_level": plot_level(df.iloc[te], p)}
            print(f"[P3] {name:18s} R2 {results['P3_temporal_2022_to_2023'][name]['image_level']['R2']:.3f}", flush=True)

    # ---------------- select best model on P1 RMSE (validation), never on P2/P3
    cand = {n: r["image_level"]["RMSE"] for n, r in results["P1_grouped_cv"].items() if n != "Mean baseline"}
    best = min(cand, key=cand.get)
    results["selected_model"] = best
    results["selection_rule"] = "lowest image-level RMSE under P1 grouped 5-fold CV"
    resid = np.abs(oof[f"P1_{best}"] - y)
    q90 = float(np.quantile(resid, 0.9))
    results["conformal_q90_t_ha"] = q90
    results["conformal_coverage_check"] = float(np.mean(resid <= q90))
    print("selected:", best, "conformal 90% half-width:", round(q90, 3))

    # ---------------- final model on all data (tuned with grouped inner CV)
    final, params = fit_tuned(best, X, y, block_groups)
    results["final_params"] = params
    # permutation importance on held-out data: model fit without one block-fold, evaluated on it
    tr, te = next(GroupKFold(n_splits=n_folds).split(X, y, block_groups))
    m_imp, _ = fit_tuned(best, X.iloc[tr], y.iloc[tr], block_groups.iloc[tr])
    pi = permutation_importance(m_imp, X.iloc[te], y.iloc[te], n_repeats=5, random_state=config.SEED,
                                scoring="neg_root_mean_squared_error", n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
    results["permutation_importance"] = imp.round(5).to_dict()

    ranges = {c: (float(X[c].quantile(0.01)), float(X[c].quantile(0.99))) for c in feats}
    bundle = YieldBundle(track=a.track, model_name=best, pipeline=final, features=feats, interval_q90=q90,
                         cv_metrics=results["P1_grouped_cv"][best], feature_ranges=ranges,
                         feature_importance=imp.head(20).round(5).to_dict(),
                         training_summary={"rows": len(df), "plots": int(df.plot_uid.nunique()),
                                           "site_seasons": sorted(seasons.unique()),
                                           "yield_mean": float(y.mean()), "yield_sd": float(y.std()),
                                           "dataset": "Best Rice (Zenodo 10.5281/zenodo.18768633)"},
                         feature_mode=a.feature_mode)
    joblib.dump(bundle, config.MODELS_DIR / f"yield_{a.track}{a.tag}.joblib")
    oof.to_csv(config.OUTPUTS_DIR / f"yield_predictions_{a.track}{a.tag}.csv", index=False)
    results["train_seconds"] = round(time.time() - t0, 1)
    (config.OUTPUTS_DIR / f"yield_results_{a.track}{a.tag}.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"saved models/yield_{a.track}{a.tag}.joblib  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
