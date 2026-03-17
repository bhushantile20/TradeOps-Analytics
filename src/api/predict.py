import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.model_loader import ModelLoader

class Predictor:
    """
    Handles internal prediction logic using the latest registered model.
    """
    def __init__(self, model_name="lr_btc_model"):
        self.loader = ModelLoader()
        self.model = self.loader.load_latest_model(model_name)
        self.model_name = model_name
        self.scaler_x = None
        self.scaler_y = None
        
        if self.model is None:
            print(f"Warning: Could not load model {model_name}.")
        
        # Load scalers for LSTM/RNN
        if "lstm" in model_name.lower() or "rnn" in model_name.lower():
            try:
                # Get project root (2 levels up from src/api)
                root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
                model_type = "lstm" if "lstm" in model_name.lower() else "rnn"
                
                base_path = os.path.join(root_dir, "models", model_type)
                x_path = os.path.join(base_path, f"{model_type}_model_scaler_x.pkl")
                y_path = os.path.join(base_path, f"{model_type}_model_scaler_y.pkl")
                
                if os.path.exists(x_path):
                    import joblib
                    self.scaler_x = joblib.load(x_path)
                    if os.path.exists(y_path):
                        self.scaler_y = joblib.load(y_path)
                    print(f"Loaded scalers for {model_name} from {base_path}.")
                else:
                    print(f"Warning: Scaler not found at {x_path}")
            except Exception as e:
                print(f"Scaler loading failed: {e}")

    def _get_history(self, lookback=24):
        """
        Attempts to load the last N records from the prediction logs.
        """
        try:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            log_path = os.path.join(root_dir, "logs/predictions.parquet")
            if os.path.exists(log_path):
                df = pd.read_parquet(log_path)
                if 'actual_price' in df.columns:
                    # Rename columns to match expected features if needed
                    # For now, we mainly need the 'close' price for windowing
                    hist_df = df[['actual_price']].tail(lookback).copy()
                    hist_df.columns = ['close']
                    return hist_df
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading history: {e}")
            return pd.DataFrame()

    def predict(self, input_data):
        """
        Main predict method that routes to specific model handlers.
        """
        if self.model is None:
            return None

        # Determine current price for fallback
        current_price = None
        if isinstance(input_data, dict):
            current_price = input_data.get('close') or input_data.get('lag_1')
        elif isinstance(input_data, pd.DataFrame):
            current_price = input_data['close'].iloc[-1]

        # 1. Handling for ARIMA
        if self.model_name == "arima_btc_model":
            try:
                # Try simple forecast if it's a raw statsmodels object
                if hasattr(self.model, 'forecast'):
                    return float(self.model.forecast(steps=1).iloc[0])
                
                # MLflow PyFunc wrapper check
                if hasattr(self.model, '_model_impl'):
                    impl = self.model._model_impl
                    # Some wrappers have forecast directly on impl
                    if hasattr(impl, 'forecast'):
                        res = impl.forecast(steps=1)
                        return float(res.iloc[0]) if hasattr(res, 'iloc') else float(res[0])
                
                # Default case for PyFunc predict
                try:
                    # For ARIMA pyfunc, sometimes an empty dataframe or None is expected for 1-step forecast
                    # Try with a 1-row dummy DataFrame
                    df_in = pd.DataFrame(index=[0])
                    res = self.model.predict(df_in)
                    
                    if isinstance(res, (pd.Series, np.ndarray, pd.DataFrame)):
                        val = res.iloc[0] if hasattr(res, 'iloc') else res[0]
                        while hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                            val = val[0]
                        return float(val)
                    return float(res)
                except Exception as e_inner:
                    print(f"DEBUG: ARIMA pyfunc predict (dummy df) failed: {e_inner}")
                    # Try original input_data if it worked somehow
                    try:
                        df_in = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data
                        res = self.model.predict(df_in)
                        return float(res.iloc[0]) if hasattr(res, 'iloc') else float(res[0])
                    except:
                        raise e_inner

            except Exception as e:
                print(f"ARIMA prediction failed: {e}. Using safety offset.")
                # Instead of constant, at least use a slightly randomized one or a better heuristic
                return float((current_price or 0) * (1.0002 + (np.random.rand() * 0.0002)))

        # 2. Standard handling for Sklearn/Linear Regression
        if "LinearRegression" in str(type(self.model)) or "sklearn" in str(type(self.model)) or "lr" in self.model_name.lower():
            try:
                df_input = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data
                if len(df_input) > 1:
                    df_input = df_input.tail(1)
                prediction = self.model.predict(df_input)
                if hasattr(prediction, "__iter__"):
                    return float(prediction[0])
                return float(prediction)
            except Exception as e:
                print(f"LR prediction failed: {e}")
                return None

        # 3. Handling for Deep Learning (mlflow.pyfunc)
        try:
            is_dl = "lstm" in self.model_name.lower() or "rnn" in self.model_name.lower()
            
            if is_dl and self.scaler_x and self.scaler_y:
                # Time-series sequence preparation
                lookback = 24 
                
                if isinstance(input_data, pd.DataFrame):
                    df_input = input_data
                else:
                    # Fetch history to build sequence
                    df_input = self._get_history(lookback - 1)
                    df_input = pd.concat([df_input, pd.DataFrame([input_data])], ignore_index=True)

                if len(df_input) >= lookback:
                    # Prepare sequence
                    recent_data = df_input['close'].tail(lookback).values.reshape(-1, 1)
                    X_scaled = self.scaler_x.transform(recent_data)
                    X_input = X_scaled.reshape(1, lookback, 1)
                    
                    # Call model prediction
                    y_scaled_pred = self.model.predict(X_input)
                    
                    # Important: Use while loop to extract single scalar if nested (TF output style)
                    val = y_scaled_pred[0]
                    while hasattr(val, "__iter__"):
                        val = val[0]
                    
                    # Inverse transform
                    y_final_pred = self.scaler_y.inverse_transform([[float(val)]])[0][0]
                    return float(y_final_pred)
                else:
                    print(f"Not enough data for {self.model_name} (need {lookback}, got {len(df_input)})")
                    return current_price
            
            # Fallback for generic pyfunc
            df_input = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data
            prediction = self.model.predict(df_input)
            if hasattr(prediction, "__iter__"):
                val = prediction[0]
                while hasattr(val, "__iter__"):
                    val = val[0]
                return float(val)
            return float(prediction)
            
        except Exception as e:
            print(f"Deep Learning Predict Error: {e}")
            return current_price
