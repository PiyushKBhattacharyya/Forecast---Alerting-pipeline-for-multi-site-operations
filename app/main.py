"""
References:
- FastAPI documentation: https://fastapi.tiangolo.com/
- Python pathlib documentation: https://docs.python.org/3/library/pathlib.html
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path
import tempfile
import zipfile

from src.pipeline import run_pipeline

# Initialize app
app = FastAPI(title="Forecast + Anomaly Detection API")

# Paths
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Output"

# API Endpoints
@app.get("/run")
def run_pipeline_endpoint():
    """
    Trigger the full pipeline:
    - Loads raw data
    - Prepares features
    - Runs forecasts (units + power)
    - Detects anomalies
    - Saves CSV outputs under OUTPUT_DIR
    """
    run_pipeline()
    return {"status": "Pipeline executed successfully"}


@app.get("/download/all")
def download_all():
    """
    Download all output files (forecasts, metrics, anomalies) bundled as a single ZIP archive.
    """
    mapping = {
        "forecast_results.csv": OUTPUT_DIR / "forecast_results.csv",
        "forecast_metrics.csv": OUTPUT_DIR / "forecast_metrics.csv",
        "alerts.csv": OUTPUT_DIR / "alerts.csv",
    }

    # Check files exist
    missing = [name for name, path in mapping.items() if not path.exists()]
    if missing:
        return {"error": f"Missing files: {missing}. Run /run first."}

    # Create temporary zip
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        zip_path = Path(tmp.name)

    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, path in mapping.items():
            zf.write(path, arcname=name)

    return FileResponse(path=zip_path, filename="pipeline_outputs.zip", media_type="application/zip")


@app.get("/download/{file_type}")
def download_file(file_type: str):
    """
    Download a single processed file.

    Parameters
    ----------
    file_type : str
        Must be one of: forecasts_units, forecasts_power, metrics, alerts
    """
    mapping = {
        "forecasts": OUTPUT_DIR / "forecast_results.csv",
        "metrics": OUTPUT_DIR / "forecast_metrics.csv",
        "alerts": OUTPUT_DIR / "alerts.csv",
    }

    if file_type not in mapping:
        return {"error": f"Invalid file_type '{file_type}'. Must be one of {list(mapping.keys())}"}

    file_path = mapping[file_type]
    if not file_path.exists():
        return {"error": f"File not found: {file_path.name}. Run /run first."}

    return FileResponse(path=file_path, filename=file_path.name, media_type="text/csv")
