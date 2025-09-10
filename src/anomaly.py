"""
anomaly.py - Detect downtime anomalies using STL decomposition and robust z-score.

References:
- STL decomposition: statsmodels.tsa.seasonal.STL
  https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html
- Z-score based anomaly detection (general statistical method).
"""

import numpy as np
import pandas as pd
from typing import List

# Interpretable anomaly detection using STL residuals + robust z-score
try:
    from statsmodels.tsa.seasonal import STL
    HAS_STL = True
except Exception:
    HAS_STL = False


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return 0.6745 * (x - med) / mad


def stl_residuals(series: pd.Series, period: int = 7) -> np.ndarray:
    """
    Compute STL residuals. If STL is unavailable or series too short, fallback to rolling mean residuals.
    """
    s = series.copy().fillna(method="ffill").fillna(method="bfill")

    # Use STL if available and series long enough
    if HAS_STL and len(s) >= period * 2:
        try:
            stl = STL(s, period=period, robust=True)
            res = stl.fit()
            residuals = s - res.trend - res.seasonal
            return residuals.values
        except Exception:
            pass

    # Fallback: rolling mean residuals
    mu = s.rolling(window=period, min_periods=1, center=True).mean()
    residuals = s - mu
    return residuals.values


def detect_anomalies(df: pd.DataFrame, targets: List[str] = ["units_produced", "power_kwh"],
                     period: int = 7, z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Detect anomalies per site and metric using STL residuals + robust z-score.
    Returns a DataFrame with site_id, date, metric, observed, expected, residual, anomaly_score, and rule.
    """
    alerts = []
    df = df.sort_values(["site_id", "date"]).reset_index(drop=True)

    for site_id, g in df.groupby("site_id"):
        for tgt in targets:
            if tgt not in g.columns:
                continue

            resid = stl_residuals(g[tgt], period=period)
            z = robust_zscore(resid)
            mask = np.abs(z) >= z_thresh

            if mask.any():
                for i in np.where(mask)[0]:
                    observed = g[tgt].iloc[i]
                    expected = observed - resid[i]
                    alerts.append({
                        "site_id": site_id,
                        "date": g["date"].iloc[i],
                        "metric": tgt,
                        "observed": float(observed),
                        "expected": float(expected),
                        "residual": float(resid[i]),
                        "anomaly_score": float(z[i]),
                        "rule": f"|z|>={z_thresh} via STL residuals",
                    })

    return pd.DataFrame(alerts)
