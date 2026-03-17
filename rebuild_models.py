import os
import sys
import subprocess

def rebuild():
    print("--- Rebuilding All BTC Forecasting Models ---")
    
    # Find project root (look for config.yaml up to 3 levels)
    root = os.getcwd()
    for _ in range(3):
        if os.path.exists(os.path.join(root, "config.yaml")):
            break
        root = os.path.dirname(root)
    
    print(f"Project Root identified as: {root}")
    
    # 1. Run the training pipeline
    print("Step 1: Running training pipeline...")
    train_script = os.path.join(root, "src/models/train.py")
    if not os.path.exists(train_script):
        # Fallback if we are in ML_Dashboard
        train_script = os.path.join(root, "..", "src/models/train.py")
        if os.path.exists(train_script):
            root = os.path.dirname(root)
        else:
            print(f"Error: Could not find train.py at {train_script}")
            return

    try:
        result = subprocess.run([sys.executable, train_script], 
                              cwd=root, 
                              capture_output=True, 
                              text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Training failed: {result.stderr}")
            return
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # 2. Verify registration in MLflow
    print("\nStep 2: Verifying MLflow registration...")
    try:
        import mlflow
        # Reading from config.yaml directly to be sure
        import yaml
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        client = mlflow.tracking.MlflowClient()
        models = [m.name for m in client.search_registered_models()]
        
        expected = ["arima_btc_model", "lr_btc_model"]
        print(f"Registered Models: {models}")
        
        missing = [m for m in expected if m not in models]
        if missing:
            print(f"WARNING: Missing models in registry: {missing}")
        else:
            print("SUCCESS: All models are registered and active.")
            
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    rebuild()
