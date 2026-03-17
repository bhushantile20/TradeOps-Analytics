import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

class PredictionMonitor:
    """
    Analyzes prediction logs for anomalies and shifts.
    """
    def __init__(self, log_path="logs/predictions.parquet"):
        self.log_path = log_path

    def analyze_predictions(self):
        if not os.path.exists(self.log_path):
            return {"status": "No logs found"}

        df = pd.read_parquet(self.log_path)
        if len(df) < 5:
            return {"status": "Insufficient data"}

        preds = df['predicted_return']
        
        # 1. Detect Outliers (Z-score > 3)
        mean_p = preds.mean()
        std_p = preds.std()
        outliers = df[np.abs((preds - mean_p) / std_p) > 3]
        
        # 2. Basic stats
        stats = {
            "timestamp": datetime.now().isoformat(),
            "count": int(len(df)),
            "mean": float(mean_p),
            "std": float(std_p),
            "anomalies_count": int(len(outliers)),
            "latest_anomalies": outliers.tail(3).to_dict('records')
        }
        
        print(f"Prediction Monitoring: Found {len(outliers)} anomalies out of {len(df)} predictions.")
        return stats

if __name__ == "__main__":
    monitor = PredictionMonitor()
    results = monitor.analyze_predictions()
    with open("reports/drift/prediction_monitoring.json", "w") as f:
        json.dump(results, f, indent=4)
