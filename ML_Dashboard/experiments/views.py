from django.shortcuts import render
import mlflow
import yaml
import os
import pandas as pd
import pytz
from datetime import datetime
from monitoring.utils import calculate_metrics

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../../config.yaml')
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_latest_runs(config):
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    experiment = mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
    runs = []
    if experiment:
        # Search latest 10 runs
        all_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id], 
            order_by=["start_time DESC"],
            max_results=10
        )
        
        ist = pytz.timezone('Asia/Kolkata')
        for _, run in all_runs.iterrows():
            start_time = run.get("start_time", None)
            formatted_time = "N/A"
            
            if start_time and hasattr(start_time, 'astimezone'):
                # Convert UTC to IST
                start_time_ist = start_time.astimezone(ist)
                formatted_time = start_time_ist.strftime('%d %b %Y, %I:%M %p')
            elif start_time:
                formatted_time = str(start_time)

            accuracy = run.get("metrics.accuracy")
            if pd.isna(accuracy):
                # Fallback to direction_accuracy for legacy runs if available
                accuracy = run.get("metrics.direction_accuracy", 0.0) * 100
                if pd.isna(accuracy):
                    accuracy = 0.0

            runs.append({
                "start_time": formatted_time,
                "run_name": run.get("tags.mlflow.runName", "N/A"),
                "model_type": run.get("params.model_type", "N/A"),
                "rmse": run.get("metrics.rmse", 0.0),
                "accuracy": accuracy,
                "status": run.get("status", "FINISHED")
            })
    return runs

def experiment_list(request):
    config = load_config()
    runs = get_latest_runs(config)
    
    # Load latest performance dynamically
    model_metrics = {}
    top_performer = "N/A"
    log_path = os.path.join(os.path.dirname(__file__), '../../logs/predictions.parquet')
    if os.path.exists(log_path):
        df = pd.read_parquet(log_path)
        raw_metrics = calculate_metrics(df)
        model_metrics = {k.lower().replace(' ', '_'): v for k, v in raw_metrics.items()}
        
        best_rmse = float('inf')
        for m, m_data in raw_metrics.items():
            if m_data['rmse'] < best_rmse and m_data['rmse'] > 0:
                best_rmse = m_data['rmse']
                top_performer = m
            
    return render(request, 'experiments/list.html', {
        'runs': runs,
        'model_metrics': model_metrics,
        'top_performer': top_performer
    })

def refresh_training_history(request):
    config = load_config()
    runs = get_latest_runs(config)
    return render(request, 'experiments/runs_table_rows.html', {'runs': runs})
