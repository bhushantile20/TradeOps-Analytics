import pandas as pd
import os
from datetime import datetime

class PredictionLogger:
    """
    Logs features and predictions to a Parquet file.
    """
    def __init__(self, log_path=None):
        if log_path is None:
            # Get project root (2 levels up from src/monitoring)
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            self.log_path = os.path.join(root_dir, "logs/predictions.parquet")
        else:
            self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        print(f"PredictionLogger initialized with log_path: {self.log_path}")

    def log_prediction(self, features: dict, forecast_timestamp=None, **predictions):
        """
        Appends new prediction entries to the log file.
        Accepts model names as keys (e.g., arima_prediction=0.1, actual=0.05).
        """
        new_entry = features.copy()
        new_entry['timestamp'] = pd.Timestamp.now(tz='Asia/Kolkata')
        
        # Explicitly store the hour we are predicting for
        if forecast_timestamp:
            new_entry['forecast_timestamp'] = pd.to_datetime(forecast_timestamp)
        else:
            # Default to +1 hour if not provided
            new_entry['forecast_timestamp'] = new_entry['timestamp'] + pd.Timedelta(hours=1)

        for key, val in predictions.items():
            new_entry[key] = float(val) if val is not None else None
        
        df_new = pd.DataFrame([new_entry])
        
        if os.path.exists(self.log_path):
            df_old = pd.read_parquet(self.log_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined.to_parquet(self.log_path, index=False)
        else:
            df_new.to_parquet(self.log_path, index=False)
        
        print(f"Prediction logged to {self.log_path}")
