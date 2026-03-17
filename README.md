 # 🚀 BTC Forecasting Pipeline & ML Dashboard

A full-stack MLOps project for real-time Bitcoin (BTC/USDT) return forecasting. This system integrates automated data ingestion, feature engineering, model training with MLflow tracking, and a dynamic monitoring dashboard.

---
## Folder Structure 

<img width="384" height="908" alt="image" src="https://github.com/user-attachments/assets/0a53f0ce-f866-4a0b-9d0f-75ff1430e999" />

<img width="340" height="330" alt="image" src="https://github.com/user-attachments/assets/2aa4ed5b-ce4a-40d2-9e86-60d4b6f69efa" />



## 📊 Dashboard Preview
 
<img width="1888" height="964" alt="image" src="https://github.com/user-attachments/assets/cc2459ca-c89f-44c1-b8c0-3246bc2883fd" />

 

---

## ✨ Key Features
- **Live Data Ingestion**: Automated fetching of OHLCV data from Binance using CCXT.
- **Dual Model Approach**:
    - **ARIMA**: Time-series analysis for trend-based forecasting.
    - **Linear Regression**: Feature-based model using technical indicators and lags.
- **Real-Time Inference**: FastAPI-powered pipeline generating hourly return forecasts.
- **Experiment Tracking**: Full integration with **MLflow** for logging parameters, metrics (RMSE, MAE, Directional Accuracy), and model versioning.
- **Monitoring & Drift**: Dynamic dashboard showing **Population Stability Index (PSI)** and KS-tests to detect data drift between training and serving data.
- **Modern UI**: Dark-themed dashboard built with Django, Tailwind CSS, and Chart.js.

---

## 🏗️ Project Structure
```text
btc_forecasting/
├── ML_Dashboard/        # Django-based monitoring dashboard
├── data/
│   ├── raw/             # Raw market data (Parquet)
│   └── processed/       # Engineered features (Parquet)
├── logs/                # Inference and prediction logs
├── mlruns/              # MLflow experiment tracking storage
├── src/
│   ├── api/             # FastAPI inference services & scheduler
│   ├── data/            # Data ingestion & automated refresh scripts
│   ├── features/        # Technical indicators (RSI, MA) & lag features
│   ├── models/          # Model training and loading logic
│   └── monitoring/      # Data drift and PSI calculation engines
├── config.yaml          # Centralized project configuration
└── requirements.txt     # Dependency list
```

---

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configure Tracking
Update `config.yaml` with your local absolute path for MLflow:
```yaml
mlflow:
  tracking_uri: file:///your/absolute/path/btc_forecasting/mlruns
```

### 3. Run the Services
**Start the Inference API (FastAPI):**
```bash
uvicorn src.api.app:app --port 8000
```

**Start the Monitoring Dashboard (Django):**
```bash
cd ML_Dashboard
python manage.py runserver 8005
```

---

## 📈 Evaluation Metrics
The system predicts **log returns** for the next hour. Performance is measured using:
- **Directional Accuracy**: The "Hit Rate" - percentage of times the model correctly predicted the price direction (Up/Down).
- **RMSE/MAE**: Measures the magnitude of prediction error in return units.

---

## 🛡️ Model Health (Drift Detection)
We use **PSI (Population Stability Index)** to compare the distribution of live inference data against the training baseline.
- **PSI < 0.1**: Healthy (Stable)
- **0.1 < PSI < 0.2**: Degraded (Monitor closely)
- **PSI > 0.2**: Drift Detected (Retraining required)

---

## 📸 More Screenshots
  ## Model Comparison

  <img width="1915" height="962" alt="image" src="https://github.com/user-attachments/assets/4ce3b254-f292-44f0-9697-3039e4e1edb2" />

 ## Data Drifting Alert

 <img width="1897" height="966" alt="image" src="https://github.com/user-attachments/assets/3f9beca0-f9fe-4a73-a547-2357c0199c8b" />

 ## Run Interface

 <img width="1890" height="903" alt="image" src="https://github.com/user-attachments/assets/d0f426dd-42c5-4c26-b10c-bf796abc8719" />

 ## Call Run Interface

 <img width="1913" height="958" alt="image" src="https://github.com/user-attachments/assets/762146d7-b3ae-4274-a2d4-9cd95955504e" />



