import pandas as pd
import numpy as np
import yaml
import mlflow
import os
import joblib
from arima_model import ARIMAModel
from linear_regression_model import LinearRegressionModel
# from lstm_model import LSTMModel
# from rnn_model import RNNModel
import sys

# Add src to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evaluation.metrics import calculate_rmse, calculate_mae, calculate_mape, calculate_accuracy, direction_accuracy

def load_config():
    """Load configuration from config.yaml."""
    paths = ["config.yaml", "../config.yaml", "../../config.yaml"]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("config.yaml not found")

def train_pipeline():
    config = load_config()
    data_path = config['data']['processed_path']
    target_col = 'next_return'
    
    # MLflow Setup
    tracking_uri = config['mlflow']['tracking_uri']
    if not (tracking_uri.startswith("http") or tracking_uri.startswith("sqlite") or tracking_uri.startswith("file")):
        tracking_uri = "file:///" + os.path.abspath(tracking_uri).replace("\\", "/")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # Load data
    if not os.path.exists(data_path):
        data_path = os.path.join("..", data_path)
    df = pd.read_parquet(data_path)
    
    # Time-series split
    split_idx = int(len(df) * config['models']['train_test_split'])
    train_df = df.iloc[:split_idx]
    
    # We define a common test_df that starts after the maximum lookback 
    # required by any model to ensure perfect alignment.
    # LSTM/RNN lookback is usually 24.
    max_lookback = 24
    test_df = df.iloc[split_idx:]
    common_test_prices = test_df['close'].values
    
    # 1. ARIMA Model
    with mlflow.start_run(run_name="ARIMA"):
        arima_order = config['models']['arima']['order']
        model = ARIMAModel(p=arima_order[0], d=arima_order[1], q=arima_order[2])
        
        # ARIMA trained on prices
        model.train(train_df['close'])
        predictions_price = model.predict(steps=len(test_df))
        
        # Metrics - Evaluate on price directly
        y_true_price = common_test_prices
        y_pred_price = predictions_price.values[:len(y_true_price)]
        
        metrics = {
            "rmse": calculate_rmse(y_true_price, y_pred_price),
            "mae": calculate_mae(y_true_price, y_pred_price),
            "mape": calculate_mape(y_true_price, y_pred_price),
            "accuracy": calculate_accuracy(y_true_price, y_pred_price),
        }
        
        # Log to MLflow
        mlflow.log_param("model_type", "ARIMA")
        mlflow.log_params({"p": arima_order[0], "d": arima_order[1], "q": arima_order[2]})
        mlflow.log_metrics(metrics)
        
        model_save_path = "models/arima/arima_model.pkl"
        model.save_model(model_save_path)
        
        mlflow.log_artifact(model_save_path)
        model_name = "arima_btc_model"
        # Use statsmodels flavor for better pyfunc compatibility
        mlflow.statsmodels.log_model(model.model_fit, "model", registered_model_name=model_name)
        
        # Transition to Staging
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["None"])
        if versions:
            latest_version = versions[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage="Staging"
            )
        
        print(f"ARIMA Metrics (Price Level): {metrics}")

    # 2. Linear Regression Model
    with mlflow.start_run(run_name="LinearRegression"):
        lr_features = config['models']['linear_regression']['features']
        model = LinearRegressionModel()
        
        X_train = train_df[lr_features]
        y_train = train_df[target_col]
        X_test = test_df[lr_features]
        
        model.train(X_train, y_train)
        pred_returns = model.predict(X_test)
        
        # Convert predicted returns back to prices
        # P_t = P_{t-1} * exp(R_t)
        # prev_prices are prices at split_idx-1, split_idx, ... split_idx + len(test_df)-2
        prev_prices = df['close'].iloc[split_idx-1 : split_idx + len(test_df)-1].values
        predictions_price = prev_prices * np.exp(pred_returns)
        
        # Metrics - Evaluate on price directly
        y_true_price = common_test_prices
        y_pred_price = predictions_price
        
        metrics = {
            "rmse": calculate_rmse(y_true_price, y_pred_price),
            "mae": calculate_mae(y_true_price, y_pred_price),
            "mape": calculate_mape(y_true_price, y_pred_price),
            "accuracy": calculate_accuracy(y_true_price, y_pred_price),
        }
        
        # Log to MLflow
        mlflow.log_param("model_type", "Linear Regression")
        mlflow.log_param("features", str(lr_features))
        mlflow.log_metrics(metrics)
        
        model_save_path = "models/linear_regression/lr_model.pkl"
        model.save_model(model_save_path)
        
        mlflow.log_artifact(model_save_path)
        model_name = "lr_btc_model"
        mlflow.sklearn.log_model(model.model, "model", registered_model_name=model_name)

        # Transition to Staging
        versions = client.get_latest_versions(model_name, stages=["None"])
        if versions:
            latest_version = versions[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage="Staging"
            )
        
        print(f"Linear Regression Metrics (Price Level): {metrics}")

    # LSTM and RNN bypassed for Python 3.14 compatibility (TensorFlow issues)
    """
    # 3. LSTM Model
    ... (omitted) ...
    """
    print("Skipping LSTM and RNN models due to environment compatibility.")


if __name__ == "__main__":
    train_pipeline()
