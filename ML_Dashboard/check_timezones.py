import os
import mlflow
import yaml
import datetime

def load_config():
    config_path = '../config.yaml'
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
experiment = mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
if experiment:
    all_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], 
        order_by=["start_time DESC"],
        max_results=5
    )
    if not all_runs.empty:
        for i, run in all_runs.iterrows():
            st = run['start_time']
            print(f"Run {i}:")
            print(f"  Start Time: {st}")
            print(f"  Type: {type(st)}")
            if hasattr(st, 'tzinfo'):
                print(f"  TZ Info: {st.tzinfo}")
    else:
        print("No runs found")
else:
    print("\nMLflow experiment not found")
