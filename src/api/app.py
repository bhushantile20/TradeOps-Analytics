from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from api.predict import Predictor
from monitoring.prediction_logger import PredictionLogger
from api.inference_pipeline import run_full_inference

app = FastAPI(title="BTCUSD Forecasting API")

# Enable CORS for cross-origin requests from the dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models and loggers
lr_predictor = Predictor(model_name="lr_btc_model")
arima_predictor = Predictor(model_name="arima_btc_model")
lstm_predictor = Predictor(model_name="lstm_btc_model")
rnn_predictor = Predictor(model_name="rnn_btc_model")
logger = PredictionLogger()

@app.post("/trigger")
def trigger_inference():
    """
    Triggers the full automated pipeline: fetch -> features -> predict -> log.
    Uses pre-loaded models for speed.
    """
    try:
        result = run_full_inference(
            lr_predictor=lr_predictor, 
            arima_predictor=arima_predictor, 
            lstm_predictor=lstm_predictor,
            rnn_predictor=rnn_predictor,
            logger=logger
        )
        if not result:
            raise HTTPException(status_code=500, detail="Automated inference failed.")
        return result
    except Exception as e:
        import traceback
        print("--- EXCEPTION IN /TRIGGER ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class PredictionInput(BaseModel):
    lag_1: float
    lag_2: float
    lag_3: float
    lag_6: float
    lag_12: float
    lag_24: float
    rolling_mean_5: float
    rolling_mean_10: float
    rolling_std_5: float
    rolling_std_10: float
    RSI: float
    moving_average_10: float
    moving_average_20: float
    volume: float

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "lr_loaded": lr_predictor.model is not None,
        "arima_loaded": arima_predictor.model is not None,
        "lstm_loaded": lstm_predictor.model is not None,
        "rnn_loaded": rnn_predictor.model is not None
    }

@app.get("/price")
def get_latest_price():
    """Returns the latest known BTC price from the prediction log for the navbar."""
    try:
        import pandas as pd
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), '../../../config.yaml')
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        log_path = os.path.join(os.path.dirname(__file__), '../../../logs/predictions.parquet')
        if os.path.exists(log_path):
            df = pd.read_parquet(log_path)
            if 'actual_price' in df.columns and not df.empty:
                latest_price = float(df['actual_price'].dropna().iloc[-1])
                return {"price": latest_price, "formatted": f"${latest_price:,.2f}"}
        return {"price": None, "formatted": "N/A"}
    except Exception as e:
        return {"price": None, "formatted": "N/A", "error": str(e)}

@app.post("/predict")
def predict(data: PredictionInput):
    if lr_predictor.model is None and arima_predictor.model is None:
        raise HTTPException(status_code=503, detail="Models not available")
    
    input_dict = data.dict()
    
    # 1. Linear Regression Prediction
    lr_pred = lr_predictor.predict(input_dict) if lr_predictor.model else None
    
    # 2. ARIMA Prediction
    arima_pred = arima_predictor.predict(input_dict) if arima_predictor.model else None
    
    # 3. LSTM Prediction
    lstm_pred = lstm_predictor.predict(input_dict) if lstm_predictor and lstm_predictor.model else None
    
    # 4. RNN Prediction
    rnn_pred = rnn_predictor.predict(input_dict) if rnn_predictor and rnn_predictor.model else None
    
    # Log predictions with current price (lag_1 is the price used for prediction)
    current_price = input_dict.get('lag_1')
    
    lr_price = current_price * np.exp(lr_pred) if (lr_pred is not None and current_price) else None
    arima_price = arima_pred 
    
    logger.log_prediction(
        input_dict, 
        lr_price=lr_price, 
        arima_price=arima_price,
        lstm_price=lstm_pred,
        rnn_price=rnn_pred,
        actual_price=current_price 
    )
    
    return {
        "lr_prediction_return": lr_pred,
        "arima_prediction_price": arima_pred,
        "lstm_prediction_price": lstm_pred,
        "rnn_prediction_price": rnn_pred,
        "current_price": current_price
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
