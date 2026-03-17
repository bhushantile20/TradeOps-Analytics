import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from api.predict import Predictor

def debug_all_models():
    print("--- Model Prediction Debugger ---")
    
    # Mock data (BTC around 70k)
    base_price = 71000
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='1h')
    prices = [base_price + np.random.normal(0, 100) for _ in range(100)]
    df = pd.DataFrame({'timestamp': dates, 'close': prices})
    
    # Feature engineering (minimal)
    df['lag_1'] = df['close'].shift(1)
    df['lag_2'] = df['close'].shift(2)
    df['lag_3'] = df['close'].shift(3)
    df['lag_6'] = df['close'].shift(6)
    df['lag_12'] = df['close'].shift(12)
    df['lag_24'] = df['close'].shift(24)
    df['RSI'] = 50 # Mock
    df['moving_average_10'] = df['close'].rolling(10).mean()
    df['moving_average_20'] = df['close'].rolling(20).mean()
    df['rolling_mean_5'] = df['close'].rolling(5).mean()
    df['rolling_mean_10'] = df['close'].rolling(10).mean()
    df['rolling_std_5'] = df['close'].rolling(5).std()
    df['rolling_std_10'] = df['close'].rolling(10).std()
    df['volume'] = 1000
    
    test_features = df.dropna().tail(1).to_dict('records')[0]
    current_p = test_features['close']
    print(f"Current Mock Price: ${current_p:.2f}")

    models = {
        "ARIMA": "arima_btc_model",
        "LR": "lr_btc_model",
        "LSTM": "lstm_btc_model",
        "RNN": "rnn_btc_model"
    }

    results = {}
    for name, m_name in models.items():
        try:
            print(f"Running {name}...")
            predictor = Predictor(model_name=m_name)
            if "lstm" in name.lower() or "rnn" in name.lower():
                pred = predictor.predict(df)
            else:
                pred = predictor.predict(test_features)
            
            # For LR, it returns return
            if name == "LR":
                ret = pred
                pred_price = current_p * np.exp(ret)
                print(f"  {name} Return: {ret:.6f} -> Price: ${pred_price:.2f}")
                results[name] = pred_price
            else:
                print(f"  {name} Price: ${pred:.2f}")
                results[name] = pred
        except Exception as e:
            print(f"  {name} Error: {e}")

    print("\n--- Final Summary ---")
    print(f"Actual: ${current_p:.2f}")
    for name, val in results.items():
        diff = val - current_p
        perc = (diff / current_p) * 100
        print(f"{name:5}: ${val:10.2f} (Diff: ${diff:7.2f}, {perc:6.3f}%)")

if __name__ == "__main__":
    debug_all_models()
