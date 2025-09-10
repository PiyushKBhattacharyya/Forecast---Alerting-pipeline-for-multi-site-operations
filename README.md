# Forecast + Alerting Pipeline (Logic Leap AI Task)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)  
[![FastAPI](https://img.shields.io/badge/FastAPI-%2332de84.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)  

## Project Overview

This project provides a **robust, end-to-end pipeline** for site-level operational forecasting and anomaly detection. It is designed to help organizations monitor and predict:

- Daily **units produced**  
- Daily **power consumption (kWh)**  

Key capabilities include:

- **Feature engineering:** Calendar features, rolling statistics, and site metadata encoding.  
- **Forecasting:** Baseline seasonal naive and Gradient Boosting/XGBoost models.  
- **Anomaly detection:** Interpretable STL residual + robust z-score method.  
- **API interface:** Run the pipeline and download results via **FastAPI** endpoints.  

---

## Requirements

- Python 3.8+  
- Libraries:

```bash
pandas
numpy
scikit-learn
statsmodels
pydantic
fastapi
uvicorn
lightgbm
xgboost (`Optional`)
```
Install dependencies via:
```bash
pip install -r requirements.txt
```

--- 

## Data Format

### Operations Data (`operations_daily.csv`)
| Column            | Type   | Description                  |
| ----------------- | ------ | ---------------------------- |
| date              | date   | Operation date               |
| site\_id          | string | Site identifier              |
| units\_produced   | float  | Units produced per day       |
| power\_kwh        | float  | Power consumption (kWh)      |
| downtime\_minutes | float  | Optional downtime in minutes |
| ...      | ...    | Additional metadata   |


### Site Metadata (`site_meta.csv`)
| Column   | Type   | Description           |
| -------- | ------ | --------------------- |
| site\_id | string | Site identifier       |
| region   | string | Site region           |
| capacity | float  | Maximum site capacity |
| ...      | ...    | Additional metadata   |

---

## Usage

1. **Start the api:  **
```bash
uvicorn app.main:app --reload
```
| Endpoint                | Method | Description                                                    |
| ----------------------- | ------ | -------------------------------------------------------------- |
| `/run`                  | GET    | Execute full pipeline                                          |
| `/download/all`         | GET    | Download all outputs as a ZIP                                  |
| `/download/{file_type}` | GET    | Download individual file (`forecasts`, `metrics`, `anomalies`) |

**Example:**
```bash
# Run pipeline
curl http://127.0.0.1:8000/run

# Download individual CSV
curl http://127.0.0.1:8000/download/forecasts

# Download all outputs as ZIP
curl http://127.0.0.1:8000/download/all
```

---

## Feature Engineering

Automatic generation of:

- **Calendar features:**
   - dow (day of week)

   - dom (day of month)

   - month, week

   - is_weekend

- **Rolling statistics:** Mean and std for windows [3, 7, 14, 28]

- **Site metadata encoding:** Convert categorical fields to numeric codes

---

## Modeling

- **Baseline:** Seasonal naive using the last 7 days

- **Improved model:** Gradient Boosting / XGBoost

- **Backtesting:** Rolling/expanding window with MAE and MAPE metrics

- **Forecast horizon:** 14 days

---

## Anomaly Detection

- STL decomposition (or fallback to rolling mean)

- Robust z-score applied to residuals

- Flags anomalies where |z-score| >= 3

- Outputs include:
| Column         | Description                     |         |   |
| -------------- | ------------------------------- | ------- | - |
| site\_id       | Site identifier                 |         |   |
| date           | Date of anomaly                 |         |   |
| metric         | `units_produced` or `power_kwh` |         |   |
| observed       | Actual value                    |         |   |
| expected       | Predicted / baseline value      |         |   |
| residual       | Observed - expected             |         |   |
| anomaly\_score |                                 | z-score |   |
| rule           | Detection rule                  |         |   |

---

## Output Files

| File                  | Description                                |
| --------------------- | ------------------------------------------ |
| forecast\_results.csv | Future predictions per site & metric       |
| forecast\_metrics.csv | MAE & MAPE for baseline & model            |
| anomalies.csv         | Detected anomalies with scores & residuals |