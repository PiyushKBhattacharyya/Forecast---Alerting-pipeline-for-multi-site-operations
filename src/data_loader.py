"""
data_loader.py - Load operations and metadata CSVs.

References:
- pandas.read_csv documentation: https://pandas.pydata.or   g/docs/reference/api/pandas.read_csv.html
"""

import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

# Expected columns in operations_daily: date, site_id, units_produced, power_kwh, downtime_minutes (optional)
# Expected columns in site_meta: site_id, region, capacity, ...

def read_operations(path: Optional[str] = None, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    REQUIRED_COLS = {"date", "site_id", "units_produced", "power_kwh"}

    def _load(p: Path) -> pd.DataFrame:
        df = pd.read_csv(p, parse_dates=["date"])  # type: ignore[arg-type]
        if not REQUIRED_COLS.issubset(df.columns):
            raise ValueError(f"{p.name} missing required columns: {REQUIRED_COLS - set(df.columns)}")
        return df.sort_values(["site_id", "date"]).reset_index(drop=True)

    if path:
        return _load(Path(path))

    candidates = sorted(data_dir.glob("operations_daily_*.csv"))
    if candidates:
        def _days(p: Path) -> int:
            name = p.stem
            token = name.split("_")[-1]
            return int(token[:-1]) if token.endswith("d") and token[:-1].isdigit() else 0
        chosen = max(candidates, key=_days)
        return _load(chosen)

    fallback = data_dir / "operations_daily.csv"
    if fallback.exists():
        return _load(fallback)

    raise FileNotFoundError("No operations_daily CSV found in Data/")

def read_site_meta(path: Optional[str] = None) -> pd.DataFrame:
    p = Path(path) if path else DATA_DIR / "site_meta.csv"
    if not p.exists():
        raise FileNotFoundError(f"site_meta not found at {p}")
    df = pd.read_csv(p)
    return df
