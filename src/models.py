"""
models.py - Baseline and improved forecasting models.

References:
- Seasonal Naive Forecasting (Hyndman & Athanasopoulos, Forecasting: Principles and Practice).
- scikit-learn GradientBoostingRegressor:
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
- XGBoost Regressor (optional):
  https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor
- Mean Absolute Percentage Error (MAPE):
  https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor

# Optional: you can switch to XGBoost/LightGBM if installed
try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

HORIZON = 15
TARGETS = ["units_produced", "power_kwh"]


def seasonal_naive_forecast(history: pd.Series, season: int = 7, horizon: int = HORIZON) -> np.ndarray:
    if len(history) == 0:
        return np.zeros(horizon)
    if len(history) < season:
        last = history.iloc[-1]
        return np.repeat(last, horizon)
    template = history.iloc[-season:]
    reps = int(np.ceil(horizon / season))
    fc = np.tile(template.values, reps)[:horizon]
    return fc


@dataclass
class ForecastResult:
    site_id: Union[str, int]
    target: str
    dates: List[pd.Timestamp]
    y_true: List[float]
    y_hat_baseline: List[float]
    y_hat_model: List[float]
    mae_baseline: float
    mae_model: float
    mape_baseline: float
    mape_model: float


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.clip(np.abs(y_true), 1e-6, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def train_improved_model(train_df: pd.DataFrame, feature_cols: List[str], target: str):
    X = train_df[feature_cols].values
    y = train_df[target].values
    if HAS_XGB:
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    else:
        model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    return model


def rolling_backtest_site(df_site: pd.DataFrame, target: str, feature_cols: List[str], horizon: int = HORIZON,
                          min_train: int = 60) -> Tuple[ForecastResult, pd.DataFrame]:
    d = df_site.sort_values("date").reset_index(drop=True)

    # Train/validation split using expanding window with last horizon as test
    if len(d) < (min_train + horizon):
        # Fallback: train on all but last horizon
        split = max(1, len(d) - horizon)
    else:
        split = len(d) - horizon

    train = d.iloc[:split].copy()
    test = d.iloc[split:].copy()

    # Baseline
    base_fc = seasonal_naive_forecast(train[target], season=7, horizon=len(test))

    # Improved model
    model = train_improved_model(train, feature_cols, target)
    y_hat_model = model.predict(test[feature_cols].values)

    mae_baseline = float(mean_absolute_error(test[target].values, base_fc))
    mae_model = float(mean_absolute_error(test[target].values, y_hat_model))
    mape_baseline = _mape(test[target].values, base_fc)
    mape_model = _mape(test[target].values, y_hat_model)

    res = ForecastResult(
        site_id=d.loc[0, "site_id"],
        target=target,
        dates=list(pd.to_datetime(test["date"])),
        y_true=list(test[target].values.astype(float)),
        y_hat_baseline=list(base_fc.astype(float)),
        y_hat_model=list(y_hat_model.astype(float)),
        mae_baseline=mae_baseline,
        mae_model=mae_model,
        mape_baseline=mape_baseline,
        mape_model=mape_model,
    )

    # Produce next-horizon forecast using all data
    final_model = train_improved_model(d, feature_cols, target)
    # Use last known row features to roll forward naively for horizon days using calendar features only
    future_dates = pd.date_range(d["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    # Recreate calendar-only features; rolling features won't be available out-of-the-box
    fut = pd.DataFrame({"date": future_dates, "site_id": d.loc[0, "site_id"]})
    from .features import add_calendar_features
    fut = add_calendar_features(fut)
    # Fill missing rolling/meta features with last known values per column
    for c in feature_cols:
        if c not in fut.columns:
            last_val = d[c].iloc[-1] if c in d.columns else 0.0
            fut[c] = last_val
    fut = fut[feature_cols]
    future_pred = final_model.predict(fut.values)

    test["y_hat_baseline"] = base_fc
    test["y_hat_model"] = y_hat_model

    return res, pd.DataFrame({
        "site_id": d.loc[0, "site_id"],
        "date": future_dates,
        "target": target,
        "y_hat": future_pred,
    })


def run_modeling(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    forecasts = []
    metrics = []
    for site_id, g in df.groupby("site_id"):
        for tgt in TARGETS:
            if tgt not in g.columns:
                continue
            res, fut = rolling_backtest_site(g, tgt, feature_cols)
            forecasts.append(fut)
            metrics.append({
                "site_id": site_id,
                "target": tgt,
                "mae_baseline": res.mae_baseline,
                "mae_model": res.mae_model,
                "mape_baseline": res.mape_baseline,
                "mape_model": res.mape_model,
            })
    return pd.concat(forecasts, ignore_index=True), pd.DataFrame(metrics)
