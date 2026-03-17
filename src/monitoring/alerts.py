import logging
import os
from datetime import datetime

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/alerts.log",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def trigger_alert(alert_type, message, severity="WARNING"):
    """
    Logs an alert to the alerts log file.
    """
    if severity == "CRITICAL":
        logging.critical(f"[{alert_type}] {message}")
    elif severity == "ERROR":
        logging.error(f"[{alert_type}] {message}")
    else:
        logging.warning(f"[{alert_type}] {message}")
    
    print(f"Alert Triggered: [{severity}] {alert_type} - {message}")

def check_for_drift_and_alert():
    import json
    
    # 1. Check Data Drift
    if os.path.exists("reports/drift/data_drift_report.json"):
        with open("reports/drift/data_drift_report.json", "r") as f:
            data = json.load(f)
            if data.get("drift_detected"):
                trigger_alert("DATA_DRIFT", "Distribution shift detected in input features.", severity="ERROR")
                
    # 2. Check Model Drift
    if os.path.exists("reports/drift/model_drift_report.json"):
        with open("reports/drift/model_drift_report.json", "r") as f:
            data = json.load(f)
            if data.get("drift_detected"):
                trigger_alert("MODEL_DRIFT", f"Performance degradation: {', '.join(data.get('reasons', []))}", severity="CRITICAL")

if __name__ == "__main__":
    check_for_drift_and_alert()
    # Example direct call
    # trigger_alert("SYSTEM_START", "Drift detection monitoring service initialized.")
