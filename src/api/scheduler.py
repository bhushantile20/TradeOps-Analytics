import time
import requests
import schedule
from datetime import datetime, timedelta
import sys
import os

def trigger_prediction():
    print(f"[{datetime.now()}] Triggering automated inference...")
    try:
        # Use the local FastAPI endpoint
        response = requests.post("http://127.0.0.1:8001/trigger", timeout=60)
        if response.status_code == 200:
            print(f"[{datetime.now()}] Success: {response.json().get('status')}")
        else:
            print(f"[{datetime.now()}] Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[{datetime.now()}] Error: {e}")

def run_scheduler():
    print("--- Starting BTC Forecasting Scheduler ---")
    print("Inference will be triggered every hour at :01 to allow for data settlement.")
    
    # Schedule to run at the 1st minute of every hour
    schedule.every().hour.at(":01").do(trigger_prediction)
    
    # Run once immediately to catch up
    trigger_prediction()
    
    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == "__main__":
    # Check if we need to install schedule
    try:
        import schedule
    except ImportError:
        print("Installing 'schedule' library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "schedule"])
        import schedule

    run_scheduler()
