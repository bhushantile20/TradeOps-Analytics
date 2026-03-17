import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import json
import os
import yaml
from datetime import datetime

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def calculate_psi(expected, actual, buckets=10):
    """
    Calculates Population Stability Index (PSI) using quantile-based binning.
    Formula: PSI = Σ (Actual% - Expected%) * ln(Actual% / Expected%)
    """
    try:
        # Use quantiles from expected distribution to define bins
        breakpoints = np.percentile(expected, np.arange(0, 101, 100 / buckets))
        # Ensure breakpoints are unique to avoid overlap
        breakpoints = np.unique(breakpoints)
        
        if len(breakpoints) < 2:
            return 0.0
            
        # Handle outliers by adding infinite boundaries
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # Calculate counts in each bin
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]
        
        # Normalize to percentages
        expected_percents = expected_counts / len(expected)
        actual_percents = actual_counts / len(actual)
        
        # Use epsilon to avoid log(0) and division by 0
        epsilon = 1e-6
        expected_percents = np.where(expected_percents == 0, epsilon, expected_percents)
        actual_percents = np.where(actual_percents == 0, epsilon, actual_percents)
        
        # Re-normalize after epsilon adjustment
        expected_percents /= np.sum(expected_percents)
        actual_percents /= np.sum(actual_percents)
        
        # Calculate PSI
        # (A - E) * ln(A / E) is standard stable form
        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return float(psi_value)
    except Exception as e:
        print(f"Error calculating PSI: {e}")
        return 0.0

def detect_drift():
    config = load_config()
    baseline_path = config['data']['processed_path']
    serving_path = "logs/predictions.parquet"
    
    if not os.path.exists(serving_path):
        print("No serving data found yet. Skipping drift detection.")
        return

    df_baseline = pd.read_parquet(baseline_path)
    df_serving = pd.read_parquet(serving_path)
    
    features = config['models']['linear_regression']['features']
    report = {
        "timestamp": pd.Timestamp.now(tz='Asia/Kolkata').isoformat(),
        "drift_detected": False,
        "average_psi": 0.0,
        "features": {}
    }

    psi_scores = []
    for feat in features:
        if feat not in df_serving.columns or feat not in df_baseline.columns:
            continue
            
        # 1. KS Test (p-value < 0.05 indicates drift)
        ks_stat, p_value = ks_2samp(df_baseline[feat], df_serving[feat])
        
        # 2. PSI Calculation
        psi_score = calculate_psi(df_baseline[feat], df_serving[feat])
        psi_scores.append(psi_score)
        
        # Individual drift status (for detailed view)
        feat_drift_status = "No Drift"
        if psi_score > 0.2:
            feat_drift_status = "Significant Drift"
        elif psi_score > 0.1:
            feat_drift_status = "Moderate Drift"
            
        report["features"][feat] = {
            "feature_name": feat,
            "ks_pvalue": float(p_value),
            "psi_score": float(psi_score),
            "drift_status": feat_drift_status
        }

    if psi_scores:
        avg_psi = np.mean(psi_scores)
        report["average_psi"] = float(avg_psi)
        # Final alert based on average PSI > drift_threshold
        threshold = config.get('monitoring', {}).get('drift_threshold', 0.2)
        if avg_psi > threshold:
            report["drift_detected"] = True

    os.makedirs("reports/drift", exist_ok=True)
    report_path = "reports/drift/data_drift_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Data drift report saved to {report_path}. Avg PSI: {report['average_psi']:.4f}, Drift detected: {report['drift_detected']}")

if __name__ == "__main__":
    detect_drift()
