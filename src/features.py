"""
features.py - Feature engineering for site-level operations.

References:
- Rolling window features: pandas.Series.rolling
  https://pandas.pydata.org/docs/reference/api/pandas.Series.rolling.html
- Date/time feature extraction (dayofweek, month, etc.) from pandas.DatetimeIndex.
"""

import pandas as pd
import numpy as np
from typing import Optional, List

# ============================================================
# Feature engineering utilities:
# 1. Calendar features (day of week, month, etc.)
# 2. Rolling mean/std features (short- and long-term trends)
# 3. Site metadata join (attach site-level attributes)
# ============================================================

def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Add calendar/time-based features from the date column.
    - day of week, day of month, month, ISO week, weekend flag
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col]) # ensure datetime type
    d["dow"] = d[date_col].dt.dayofweek # 0=Mon, 6=Sun
    d["dom"] = d[date_col].dt.day # day of month
    d["month"] = d[date_col].dt.month # numeric month
    d["week"] = d[date_col].dt.isocalendar().week.astype(int) # ISO week number
    d["is_weekend"] = (d["dow"] >= 5).astype(int) # weekend indicator
    return d


def add_rolling_features(df: pd.DataFrame,
                         by: List[str] = ["site_id"],
                         date_col: str = "date",
                         targets: List[str] = ["units_produced", "power_kwh"],
                         windows: List[int] = [3, 7, 14, 28]) -> pd.DataFrame:
    """
    Add rolling mean and std features for target variables.
    - Groups by `site_id` to compute rolling stats per site
    - Supports multiple window sizes (default: 3, 7, 14, 28 days)
    - Creates new columns: <target>_rollmean_<window>, <target>_rollstd_<window>
    """
    d = df.copy()
    d = d.sort_values(by + [date_col]) # sort by site and date for rolling
    for tgt in targets:
        if tgt not in d.columns:
            continue
        for w in windows:
            # Rolling mean with a minimum number of observations = half the window
            d[f"{tgt}_rollmean_{w}"] = (
                d.groupby(by)[tgt]
                .transform(lambda s: s.rolling(w, min_periods=max(1, w//2)).mean())
            )
            # Rolling std with the same logic
            d[f"{tgt}_rollstd_{w}"] = (
                d.groupby(by)[tgt]
                .transform(lambda s: s.rolling(w, min_periods=max(1, w//2)).std())
            )
    return d


def join_site_meta(ops: pd.DataFrame, site_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Join site-level metadata (static features like location, capacity).
    - Ensures site_id is string in both datasets
    - Encodes categorical metadata columns as numeric codes
    """
    meta = site_meta.copy()

    # Normalize join key dtype
    ops["site_id"] = ops["site_id"].astype(str)
    meta["site_id"] = meta["site_id"].astype(str)

    # Encode categoricals except join key
    for c in meta.select_dtypes(include=["object"]).columns:
        if c != "site_id":
            try:
                meta[c] = meta[c].astype("category").cat.codes
            except Exception:
                pass

    return ops.merge(meta, on="site_id", how="left")


def prepare_features(ops: pd.DataFrame, site_meta: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Master function:
    1. Add calendar features
    2. Add rolling features
    3. Join with site metadata (if provided)
    Returns enriched dataset ready for modeling.
    """
    d = add_calendar_features(ops)
    d = add_rolling_features(d)
    if site_meta is not None:
        d = join_site_meta(d, site_meta)
    return d
