import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath('src'))
from api.predict import Predictor

def test_arima():
    print("Testing ARIMA Predictor...")
    try:
        arima_predictor = Predictor(model_name="arima_btc_model")
        print(f"Model loaded: {type(arima_predictor.model)}")
        
        # Test with dummy data
        dummy_input = {'close': 71000.0, 'lag_1': 70950.0}
        pred = arima_predictor.predict(dummy_input)
        print(f"Prediction for {dummy_input}: {pred}")
        
    except Exception as e:
        print(f"ARIMA Test Failed: {e}")
        import traceback
        traceback.print_exc()

def test_dl_scaling():
    print("\nTesting LSTM Scaling...")
    try:
        lstm_predictor = Predictor(model_name="lstm_btc_model")
        if lstm_predictor.scaler_x and lstm_predictor.scaler_y:
            print("Scalers loaded successfully.")
        else:
            print("Error: Scalers NOT loaded.")
            
        # Create history (24 rows)
        history = pd.DataFrame({
            'close': np.linspace(70000, 71000, 24)
        })
        pred = lstm_predictor.predict(history)
        print(f"Prediction with history: {pred}")
        
        # Test with 1 row (should fall back and return small value)
        one_row = {'close': 71000.0}
        pred_small = lstm_predictor.predict(one_row)
        print(f"Prediction with 1 row: {pred_small}")
        
    except Exception as e:
        print(f"DL Test Failed: {e}")

if __name__ == "__main__":
    test_arima()
    test_dl_scaling()
