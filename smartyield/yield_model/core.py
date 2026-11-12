"""Yield regression: model zoo, metrics, grouped validation and the saved
model bundle used by the API.

Why these models?
* Linear Regression: an interpretable baseline. It assumes yield is a
  weighted sum of the features.
* Random Forest: an average of many decision trees. It captures non-linear
  effects (e.g. NDVI saturation) and interactions. It is robust to unscaled
  and correlated features.
* SVR (RBF kernel): kernel regression with an epsilon-insensitive loss, so
  small noisy errors are ignored. Works well on small-to-medium tabular data.
* XGBoost: gradient-boosted trees, where each tree corrects the errors of the
  previous ones. Usually the strongest model on tabular features.
* Ridge Regression (extra): linear regression with an L2 penalty. It shows
  how much of Linear Regression's error comes from collinear features.
* Mean baseline (DummyRegressor): predicts the training mean. A model is only
  useful if it beats this.

Every model is wrapped in a Pipeline (median imputation -> scaling -> model),
so preprocessing is fit only on training folds. This prevents leakage through
imputation or scaling.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from .. import config

MODEL_NAMES = ["Mean baseline", "Linear Regression", "Ridge Regression", "Random Forest", "SVR", "XGBoost"]


def make_model(name: str) -> tuple[Pipeline, dict]:
    """Return (pipeline, hyper-parameter grid for inner-CV tuning)."""
    imp = ("impute", SimpleImputer(strategy="median"))
    if name == "Mean baseline":
        return Pipeline([imp, ("model", DummyRegressor())]), {}
    if name == "Linear Regression":
        return Pipeline([imp, ("scale", StandardScaler()), ("model", LinearRegression())]), {}
    if name == "Ridge Regression":
        # extra comparison model: linear regression + L2 penalty, which stays stable with
        # correlated features (many vegetation indices are strongly collinear)
        return (Pipeline([imp, ("scale", StandardScaler()), ("model", Ridge())]),
                {"model__alpha": [1.0, 30.0, 300.0]})
    if name == "Random Forest":
        return (Pipeline([imp, ("model", RandomForestRegressor(n_estimators=250, max_features=0.33, n_jobs=-1,
                                                               random_state=config.SEED))]),
                {"model__min_samples_leaf": [3, 12]})
    if name == "SVR":
        return (Pipeline([imp, ("scale", StandardScaler()), ("model", SVR(kernel="rbf", epsilon=0.2))]),
                {"model__C": [1.0, 4.0]})
    if name == "XGBoost":
        return (Pipeline([imp, ("model", XGBRegressor(n_estimators=350, learning_rate=0.04, subsample=0.8,
                                                      colsample_bytree=0.7, min_child_weight=5, n_jobs=-1,
                                                      tree_method="hist", random_state=config.SEED))]),
                {"model__max_depth": [3, 6]})
    raise ValueError(name)


def metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        # MAPE is well defined here: yields were filtered to > 0 t/ha (smallest kept ~0.4 t/ha)
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "n": int(len(y_true)),
    }


def add_relative_features(df: pd.DataFrame, cols: list[str], group: pd.Series | None) -> pd.DataFrame:
    """Add '<col>_rel' = value minus the median of that column over the same image.

    Within one drone image every region shares the illumination, sensor
    calibration and flight conditions. Subtracting the image median removes
    those flight-level offsets and keeps the within-field contrast that tracks
    yield differences. group=None treats the whole frame as one image (app use).
    """
    base = [c for c in cols if c != "doy" and not c.endswith("_rel")]
    out = df.copy()
    if group is None:
        med = df[base].median()
        rel = df[base] - med
    else:
        rel = df[base] - df.groupby(group)[base].transform("median")
    rel.columns = [f"{c}_rel" for c in base]
    return pd.concat([out, rel], axis=1)


@dataclass
class YieldBundle:
    """Everything the API needs to make and explain a prediction."""
    track: str                      # 'ms' or 'rgb'
    model_name: str
    pipeline: Pipeline
    features: list[str]
    unit: str = "t/ha"
    crop: str = "rice"
    interval_q90: float = float("nan")      # split-conformal half-width (90 %)
    cv_metrics: dict = field(default_factory=dict)
    feature_ranges: dict = field(default_factory=dict)
    feature_importance: dict = field(default_factory=dict)
    training_summary: dict = field(default_factory=dict)
    feature_mode: str = "abs"               # 'abs' or 'absrel' (absolute + image-relative features)

    def predict_frame(self, feats: pd.DataFrame, crop_mask: pd.Series | None = None) -> pd.DataFrame:
        """Predict per region. With feature_mode='absrel' the image medians are
        taken over crop regions only (crop_mask), matching training, where every
        region was a crop plot."""
        if self.feature_mode == "absrel":
            base = [c[:-4] for c in self.features if c.endswith("_rel")]
            ref = feats[crop_mask] if crop_mask is not None and crop_mask.any() else feats
            med = ref.reindex(columns=base).median()
            rel = feats.reindex(columns=base) - med
            rel.columns = [f"{c}_rel" for c in base]
            feats = pd.concat([feats.drop(columns=[c for c in rel.columns if c in feats]), rel], axis=1)
        X = feats.reindex(columns=self.features).astype(float)
        pred = self.pipeline.predict(X)
        lo, hi = config.CROPS[self.crop]["typical_range_t_ha"]
        pred = np.clip(pred, 0, hi * 1.5)
        out = pd.DataFrame({"pred_t_ha": pred})
        if np.isfinite(self.interval_q90):
            out["lo90_t_ha"] = np.clip(pred - self.interval_q90, 0, None)
            out["hi90_t_ha"] = pred + self.interval_q90
        # out-of-distribution warning: share of features outside the training 1-99 % range
        ood = np.zeros(len(X))
        for c, (a, b) in self.feature_ranges.items():
            if c in X:
                v = X[c].to_numpy()
                ood += ((v < a) | (v > b)) & np.isfinite(v)
        out["ood_fraction"] = ood / max(len(self.feature_ranges), 1)
        return out
