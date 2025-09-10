"""
pipeline.py - Orchestrates full pipeline: load, feature, model, anomalies, save outputs.

References:
- pathlib, pandas, and numpy standard usage.
- General pipeline structure inspired by common ML engineering practices
  (scikit-learn pipelines and modular code organization).
"""

import pandas as pd
from pathlib import Path

from src.data_loader import read_operations, read_site_meta
from src.features import prepare_features
from src.models import run_modeling, TARGETS
from src.anomaly import detect_anomalies

# Directories
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline() -> None:
    # 1. Load raw data
    ops = read_operations()
    site_meta = read_site_meta()

    # 2. Feature engineering
    feats = prepare_features(ops, site_meta=site_meta)

    # Decide which columns are usable features
    exclude = ["date", "site_id"] + TARGETS
    feature_cols = [c for c in feats.columns if c not in exclude]

    # 3. Forecasting
    fut_df, metrics_df = run_modeling(feats, feature_cols)

    # Split into required outputs
    units_fc = fut_df[fut_df["target"] == "units_produced"].drop(columns=["target"])
    power_fc = fut_df[fut_df["target"] == "power_kwh"].drop(columns=["target"])

    units_path = OUTPUT_DIR / "forecast_units.csv"
    power_path = OUTPUT_DIR / "forecast_power.csv"
    metrics_path = OUTPUT_DIR / "forecast_metrics.csv"

    units_fc.to_csv(units_path, index=False)
    power_fc.to_csv(power_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    print(f"[INFO] Units forecast saved to {units_path}")
    print(f"[INFO] Power forecast saved to {power_path}")
    print(f"[INFO] Metrics saved to {metrics_path}")

    # 4. Anomaly detection
    anomalies_df = detect_anomalies(ops, targets=TARGETS)
    anomalies_path = OUTPUT_DIR / "alerts.csv"
    anomalies_df.to_csv(anomalies_path, index=False)
    print(f"[INFO] Anomalies saved to {anomalies_path}")


if __name__ == "__main__":
    run_pipeline()