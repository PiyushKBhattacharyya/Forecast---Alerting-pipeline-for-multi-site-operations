import pandas as pd
import numpy as np
from typing import Optional, List

# Feature engineering utilities
# - Calendar features
# - Rolling means as baselines
# - Site metadata join

def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d["dow"] = d[date_col].dt.dayofweek  # 0=Mon
    d["dom"] = d[date_col].dt.day
    d["month"] = d[date_col].dt.month
    d["week"] = d[date_col].dt.isocalendar().week.astype(int)
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    return d


def add_rolling_features(df: pd.DataFrame,
                         by: List[str] = ["site_id"],
                         date_col: str = "date",
                         targets: List[str] = ["units_produced", "power_kwh"],
                         windows: List[int] = [3, 7, 14, 28]) -> pd.DataFrame:
    d = df.copy()
    d = d.sort_values(by + [date_col])
    for tgt in targets:
        if tgt not in d.columns:
            continue
        for w in windows:
            d[f"{tgt}_rollmean_{w}"] = (
                d.groupby(by)[tgt]
                .transform(lambda s: s.rolling(w, min_periods=max(1, w//2)).mean())
            )
            d[f"{tgt}_rollstd_{w}"] = (
                d.groupby(by)[tgt]
                .transform(lambda s: s.rolling(w, min_periods=max(1, w//2)).std())
            )
    return d


def join_site_meta(ops: pd.DataFrame, site_meta: pd.DataFrame) -> pd.DataFrame:
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
    d = add_calendar_features(ops)
    d = add_rolling_features(d)
    if site_meta is not None:
        d = join_site_meta(d, site_meta)
    return d
