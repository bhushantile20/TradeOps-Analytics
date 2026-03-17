import pandas as pd
import numpy as np
import yaml
import json
import os
import sys
import joblib

# Add src to sys.path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.arima_model import ARIMAModel
from models.linear_regression_model import LinearRegressionModel
from evaluation.metrics import calculate_rmse, calculate_mae, calculate_mape, direction_accuracy

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../../config.yaml')
    if not os.path.exists(config_path):
        # Try local path if called from different directory
        config_path = "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate():
    print("Starting Model Evaluation and Comparison...")
    config = load_config()
    
    # Identify data path
    data_path = config['data']['processed_path']
    if not os.path.exists(data_path):
        print(f"Error: Processed data not found at {data_path}")
        return

    # Load data
    df = pd.read_parquet(data_path)
    
    # Time-series split (consistent with Phase 2)
    split_idx = int(len(df) * config['models']['train_test_split'])
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    target_col = 'next_return'
    y_true = test_df[target_col].values
    
    results = {}

    # 1. Evaluate ARIMA
    arima_path = "models/arima/arima_model.pkl"
    if os.path.exists(arima_path):
        print("Evaluating ARIMA...")
        try:
            # Statsmodels ARIMA model_fit is stored
            model_fit = joblib.load(arima_path)
            
            # Perform one-step-ahead rolling forecast
            # We use the model's apply() method to "update" the model with test data 
            # and get a prediction for each step.
            # However, apply() returns a new result object. 
            # A more efficient way for point-wise metrics in evaluation:
            # We can use the 'get_prediction' or 'forecast' logic.
            # For ARIMA, we want to know what it would have predicted for each t in test_set
            # given observations up to t-1.
            
            # We'll use the 'apply' method to get the prediction results for the test data
            # this effectively runs the model over the test set one step at a time
            # Note: we need to pass the full data to get the context for lags
            full_series = pd.concat([train_df['close'], test_df['close']])
            res_updated = model_fit.apply(full_series)
            
            # The predictions for the test period start at len(train_df)
            # res_updated.predict() gives the predicted values (levels)
            all_preds = res_updated.predict()
            # Forecasts for the test set are from index len(train_df) to end
            forecast_prices = all_preds[len(train_df):]
            
            # Convert prices to returns for comparison with next_return
            # next_return[t] = log(price[t+1] / price[t])
            # Our forecast_prices[t] corresponds to the prediction for test_df.index[t] (i.e. p_t)
            # But test_df['next_return'][t] is log(p_t+1 / p_t).
            # So arima_prediction[t] should be log(forecast_p_t+1 / actual_p_t)? 
            # No, MLOps requirement: predict next return.
            # arima_return_pred[t] = log(forecast_p_t+1 / actual_p_t)
            
            actual_prices_prev = full_series.iloc[len(train_df)-1 : -1].values
            pred_returns = np.log(forecast_prices.values / actual_prices_prev)
            
            results["arima"] = {
                "MAE": float(calculate_mae(y_true, pred_returns)),
                "RMSE": float(calculate_rmse(y_true, pred_returns)),
                "Directional Accuracy": float(direction_accuracy(y_true, pred_returns))
            }
        except Exception as e:
            print(f"Error evaluating ARIMA: {e}")
    else:
        print(f"Warning: ARIMA model not found at {arima_path}")

    # 2. Evaluate Linear Regression
    lr_path = "models/linear_regression/lr_model.pkl"
    if os.path.exists(lr_path):
        print("Evaluating Linear Regression...")
        try:
            model = joblib.load(lr_path)
            lr_features = config['models']['linear_regression']['features']
            X_test = test_df[lr_features]
            
            predictions = model.predict(X_test)
            
            results["linear_regression"] = {
                "MAE": float(calculate_mae(y_true, predictions)),
                "RMSE": float(calculate_rmse(y_true, predictions)),
                "Directional Accuracy": float(direction_accuracy(y_true, predictions))
            }
        except Exception as e:
            print(f"Error evaluating Linear Regression: {e}")
    else:
        print(f"Warning: Linear Regression model not found at {lr_path}")

    # Print Summary
    print("\nModel Performance Summary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")

    # 3. Generate and Save Prediction History for Dashboard
    print("\nGenerating prediction history for dashboard...")
    history_len = min(50, len(test_df))
    history_df = test_df.tail(history_len).copy()
    
    # Store actual price
    history_df['actual_price'] = history_df['close']
    
    # Store ARIMA price
    if "arima" in results:
        # forecast_prices was already calculated for the whole test set
        history_df['arima_price'] = forecast_prices.tail(history_len).values
    else:
        history_df['arima_price'] = None
        
    if "linear_regression" in results:
        # Convert predicted returns to prices
        # Price_{t+1} = Price_t * exp(predicted_return)
        # Using the shifted close price to get the base for prediction
        base_prices = test_df['close'].shift(1).tail(history_len).values
        # If the first value is NaN (if history_len == len(test_df)), handle it
        if np.isnan(base_prices[0]):
             base_prices[0] = train_df['close'].iloc[-1]
        
        lr_returns = predictions[-history_len:]
        history_df['lr_price'] = base_prices * np.exp(lr_returns)
    else:
        history_df['lr_price'] = None

    # Standardize columns for dashboard
    lr_features = config['models']['linear_regression']['features']
    log_cols = ['timestamp', 'actual_price', 'arima_price', 'lr_price'] + lr_features
    # Ensure columns exist (e.g. they might be named slightly differently in history_df)
    log_cols = [c for c in log_cols if c in history_df.columns or c == 'timestamp']

    if 'timestamp' not in history_df.columns:
        history_df['timestamp'] = history_df.index
    
    # Save to parquet
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "predictions.parquet")
    history_df[log_cols].to_parquet(log_path, index=False)
    print(f"Prediction history saved to {log_path}")

    # Output to JSON
    output_dir = "monitoring"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_performance.json")
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    evaluate()
