import pandas as pd
try:
    import pandas_ta as ta
except ImportError:
    ta = None
import numpy as np
import os
import sys
import yaml
from datetime import datetime, timedelta

# Add src to path to import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from api.predict import Predictor
from monitoring.prediction_logger import PredictionLogger

def load_config():
    """Load configuration from config.yaml."""
    paths = ["config.yaml", "../config.yaml", "../../config.yaml"]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("config.yaml not found")

def run_full_inference(lr_predictor=None, arima_predictor=None, lstm_predictor=None, rnn_predictor=None, logger=None):
    print("--- Starting Automated Inference Pipeline ---")
    config = load_config()
    
    # 1. Fetch Latest Data
    print("Step 1: Fetching latest BTC-USD data from Binance...")
    try:
        import ccxt
        print("ccxt imported.")
        exchange = ccxt.binance({'timeout': 10000})
        print("CCXT exchange initialized.")
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe='1h', limit=100)
        print(f"Data fetched: {len(ohlcv)} rows.")
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        df = df.set_index('timestamp').sort_index()
        
        if df.empty:
            print("Error: No data retrieved.")
            return None
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

    df.columns = [c.lower() for c in df.columns]

    # 2. Feature Engineering
    print("Step 2: Generating features...")
    for lag in config['features']['lags']:
        df[f'lag_{lag}'] = df['close'].shift(lag)
    
    print("Calculating RSI...")
    if ta:
        df['RSI'] = ta.rsi(df['close'], length=config['features']['indicators']['rsi'])
    else:
        # Fallback RSI calculation (Simplified)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=config['features']['indicators']['rsi']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=config['features']['indicators']['rsi']).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    print("Calculating Moving Averages...")
    for ma in config['features']['indicators']['ma']:
        if ta:
            df[f'moving_average_{ma}'] = ta.sma(df['close'], length=ma)
        else:
            df[f'moving_average_{ma}'] = df['close'].rolling(window=ma).mean()
    
    print("Calculating Rolling Stats...")
    for window in config['features']['rolling_windows']:
        df[f'rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['close'].rolling(window=window).std()

    print("Cleaning data...")
    latest_row = df.dropna().tail(1)
    if latest_row.empty:
        print("Error: Not enough data for features.")
        return None
    
    features_dict = latest_row.to_dict('records')[0]
    current_price = features_dict.get('close')
    
    # Calculate forecast_timestamp from current system time (Asia/Kolkata)
    # Each inference run generates a prediction exactly 1 hour ahead
    current_time = pd.Timestamp.now(tz='Asia/Kolkata')
    forecast_timestamp = current_time + pd.Timedelta(hours=1)
    
    print(f"Current System Time: {current_time}")
    print(f"Target Forecast Time: {forecast_timestamp} (Data Timestamp: {latest_row.index[0]})")

    # 3. Use provided or create new objects
    if lr_predictor is None:
        print("Loading LR Predictor...")
        lr_predictor = Predictor(model_name="lr_btc_model")
    if arima_predictor is None:
        print("Loading ARIMA Predictor...")
        arima_predictor = Predictor(model_name="arima_btc_model")
    # LSTM and RNN bypassed for stability on Python 3.14
    # if lstm_predictor is None:
    #     lstm_predictor = Predictor(model_name="lstm_btc_model")
    # if rnn_predictor is None:
    #     rnn_predictor = Predictor(model_name="rnn_btc_model")
    if logger is None:
        print("Initializing Logger...")
        logger = PredictionLogger()

    # 4. Execute Predictions
    print("Step 4: Running inference...")
    try:
        # Filter features to match only what LR model expects
        lr_feature_names = config['models']['linear_regression']['features']
        lr_input_dict = {k: features_dict.get(k) for k in lr_feature_names}
        
        print("Running LR predict...")
        lr_ret_pred = lr_predictor.predict(lr_input_dict) if lr_predictor.model else None
        
        print("Running ARIMA predict...")
        arima_price_pred = arima_predictor.predict(features_dict) if arima_predictor.model else None
        
        print("Running LSTM predict...")
        # LSTM/RNN need historical data for sequences
        lstm_price_pred = lstm_predictor.predict(df) if (lstm_predictor and lstm_predictor.model) else None
        
        print("Running RNN predict...")
        rnn_price_pred = rnn_predictor.predict(df) if (rnn_predictor and rnn_predictor.model) else None
        
        print(f"DEBUG: Raw LR ret pred: {lr_ret_pred}")
        print(f"DEBUG: Raw ARIMA price pred: {arima_price_pred}")
        print(f"DEBUG: Raw LSTM price pred: {lstm_price_pred}")
        print(f"DEBUG: Raw RNN price pred: {rnn_price_pred}")

        print("Applying transformations...")
        lr_price_pred = current_price * np.exp(lr_ret_pred) if (lr_ret_pred is not None) else None
        
        # Verify if any prediction is identical to current_price (potential failure)
        for name, val in [("LR", lr_price_pred), ("ARIMA", arima_price_pred), ("LSTM", lstm_price_pred), ("RNN", rnn_price_pred)]:
            if val is not None and abs(val - current_price) < 0.0001:
                print(f"WARNING: {name} prediction is NEARLY IDENTICAL to current price!")
        
        # 5. Persistent Logging
        print("Step 5: Logging results...")
        logger.log_prediction(
            features_dict, 
            lr_price=lr_price_pred, 
            arima_price=arima_price_pred,
            lstm_price=lstm_price_pred,
            rnn_price=rnn_price_pred,
            actual_price=current_price,
            forecast_timestamp=forecast_timestamp
        )
    except Exception as e:
        import traceback
        print(f"FAILED during step 4/5 of inference: {e}")
        traceback.print_exc()
        return None
    
    print("Pipeline Success.")
    return {
        "status": "success",
        "actual": current_price,
        "lr_prediction": lr_price_pred,
        "arima_prediction": arima_price_pred,
        "lstm_prediction": lstm_price_pred,
        "rnn_prediction": rnn_price_pred
    }

if __name__ == "__main__":
    run_full_inference()
