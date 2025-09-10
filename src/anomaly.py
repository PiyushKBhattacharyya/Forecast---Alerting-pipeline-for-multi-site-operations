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

# ============================================================
# Core Idea:
#   - Decompose each site's time series into seasonal + trend + residual
#   - Use residuals (unexpected deviations) for anomaly detection
#   - Apply robust z-score (median absolute deviation) for interpretable alerts
# ============================================================

# Interpretable anomaly detection using STL residuals + robust z-score
try:
    from statsmodels.tsa.seasonal import STL
    HAS_STL = True
except Exception:
    HAS_STL = False


def robust_zscore(x: np.ndarray) -> np.ndarray:
    """
    Compute a robust z-score using median and MAD (median absolute deviation).
    This is less sensitive to outliers compared to mean/std-based z-scores.

    Formula:
        z = 0.6745 * (x - median(x)) / MAD

    Parameters
    ----------
    x : np.ndarray
        Input array (residuals)

    Returns
    -------
    np.ndarray
        Robust z-scores for each element in x
    """
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9 # avoid division by zero
    return 0.6745 * (x - med) / mad


def stl_residuals(series: pd.Series, period: int = 7) -> np.ndarray:
    """
    Compute residuals from STL decomposition (if available), otherwise fallback.

    STL (Seasonal-Trend decomposition using LOESS):
        series = trend + seasonal + residual

    Residual = what is left unexplained → potential anomaly signal

    Parameters
    ----------
    series : pd.Series
        Time series data (numeric, indexed by date ideally)
    period : int
        Seasonal cycle length (default 7 for weekly seasonality)

    Returns
    -------
    np.ndarray
        Residual values aligned with series index
    """
    s = series.copy().fillna(method="ffill").fillna(method="bfill")

    # Use STL if available and series is long enough (≥ 2 seasonal cycles)
    if HAS_STL and len(s) >= period * 2:
        try:
            stl = STL(s, period=period, robust=True)
            res = stl.fit()
            residuals = s - res.trend - res.seasonal
            return residuals.values
        except Exception:
            pass

    # Fallback: subtract rolling mean as crude trend/seasonal estimate
    mu = s.rolling(window=period, min_periods=1, center=True).mean()
    residuals = s - mu
    return residuals.values


def detect_anomalies(df: pd.DataFrame, targets: List[str] = ["units_produced", "power_kwh"],
                     period: int = 7, z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Detect anomalies for each site_id and target metric.

    Steps:
      1. For each site & target metric, compute STL residuals
      2. Standardize residuals using robust z-score
      3. Flag anomalies where |z| >= threshold
      4. Output interpretable alerts

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with at least ['site_id', 'date', targets...]
    targets : List[str]
        Which numeric columns to check for anomalies
    period : int
        Seasonal period for STL (default 7 = weekly seasonality)
    z_thresh : float
        Robust z-score threshold for flagging anomalies (default 3.0)

    Returns
    -------
    pd.DataFrame
        Alerts with columns:
        - site_id, date, metric
        - observed, expected, residual
        - anomaly_score (robust z)
        - rule (explanation string)
    """
    alerts = []
    df = df.sort_values(["site_id", "date"]).reset_index(drop=True)

    for site_id, g in df.groupby("site_id"):
        for tgt in targets:
            if tgt not in g.columns:
                continue
            
            # Compute residuals for this site/metric
            resid = stl_residuals(g[tgt], period=period)
            # Compute robust z-scores
            z = robust_zscore(resid)
            # Boolean mask of anomalies
            mask = np.abs(z) >= z_thresh

            if mask.any():
                for i in np.where(mask)[0]:
                    observed = g[tgt].iloc[i]
                    expected = observed - resid[i] # expected = observed - residual
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
