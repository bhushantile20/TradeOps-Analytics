import mlflow
import yaml
import os

def load_config():
    """Load configuration from config.yaml."""
    paths = ["config.yaml", "../config.yaml", "../../config.yaml"]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("config.yaml not found")

class ModelLoader:
    """
    Utility to load models from MLflow Model Registry.
    """
    def __init__(self):
        config = load_config()
        tracking_uri = config['mlflow']['tracking_uri']
        if not (tracking_uri.startswith("http") or tracking_uri.startswith("sqlite") or tracking_uri.startswith("file")):
            # Find the root directory to properly build absolute path
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            if tracking_uri == "mlruns":
                tracking_uri = os.path.join(root_dir, tracking_uri)
            tracking_uri = "file:///" + os.path.abspath(tracking_uri).replace("\\", "/")
        mlflow.set_tracking_uri(tracking_uri)

    def load_latest_model(self, model_name: str):
        """
        Loads the latest version of a model, preferring Production over Staging.
        """
        print(f"Loading latest version of {model_name}...")
        
        # Try Production first
        try:
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"Loaded {model_name} from Production stage.")
            return model
        except Exception:
            print(f"Production model not found for {model_name}. Fallback to Staging.")
            
        # Try Staging
        try:
            model_uri = f"models:/{model_name}/Staging"
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"Loaded {model_name} from Staging stage.")
            return model
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
