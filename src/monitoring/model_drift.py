import yaml
import mlflow
import json
import os
from datetime import datetime

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def detect_model_drift(thresholds=None):
    """
    Compares latest run metrics with historical averages to detect degradation.
    """
    config = load_config()
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    # Default thresholds for degradation
    if thresholds is None:
        thresholds = {
            "rmse_increase_pct": 0.2, # 20% increase
            "accuracy_decrease_pct": 0.1 # 10% decrease
        }

    experiment = mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
    if not experiment:
        return

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
    if len(runs) < 2:
        print("Not enough model runs to detect drift.")
        return

    latest_run = runs.iloc[0]
    historical_runs = runs.iloc[1:]
    
    # For simplicity, we compare with the mean of historical runs
    hist_rmse = historical_runs['metrics.rmse'].mean()
    hist_acc = historical_runs['metrics.direction_accuracy'].mean()
    
    curr_rmse = latest_run['metrics.rmse']
    curr_acc = latest_run['metrics.direction_accuracy']
    
    drift_detected = False
    reasons = []
    
    if curr_rmse > hist_rmse * (1 + thresholds["rmse_increase_pct"]):
        drift_detected = True
        reasons.append(f"RMSE increased from {hist_rmse:.6f} to {curr_rmse:.6f}")
        
    if curr_acc < hist_acc * (1 - thresholds["accuracy_decrease_pct"]):
        drift_detected = True
        reasons.append(f"Accuracy decreased from {hist_acc:.4f} to {curr_acc:.4f}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "drift_detected": drift_detected,
        "metrics": {
            "current_rmse": float(curr_rmse),
            "baseline_rmse": float(hist_rmse),
            "current_accuracy": float(curr_acc),
            "baseline_accuracy": float(hist_acc)
        },
        "reasons": reasons
    }

    report_path = "reports/drift/model_drift_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Model drift report saved to {report_path}. Drift detected: {drift_detected}")

if __name__ == "__main__":
    detect_model_drift()
