import pandas as pd
import joblib
import os
from statsmodels.tsa.arima.model import ARIMA

class ARIMAModel:
    """
    ARIMA model implementation using statsmodels.
    """
    def __init__(self, p=5, d=1, q=0):
        self.order = (p, d, q)
        self.model_fit = None

    def train(self, data: pd.Series):
        """
        Trains the ARIMA model on the provided time series.
        Note: ARIMA in statsmodels doesn't separate fit/predict as clearly 
        as sklearn for forecasting into the future if just fitting once.
        """
        print(f"Training ARIMA model with order {self.order}...")
        model = ARIMA(data, order=self.order)
        self.model_fit = model.fit()

    def predict(self, steps: int):
        """
        Forecasts future values.
        """
        if self.model_fit is None:
            raise Exception("Model must be trained before prediction.")
        
        forecast = self.model_fit.forecast(steps=steps)
        return forecast

    def save_model(self, path: str):
        """
        Saves the fitted model to a file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model_fit, path)
        print(f"ARIMA model saved to {path}")

    def load_model(self, path: str):
        """
        Loads a fitted model from a file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        self.model_fit = joblib.load(path)
        print(f"ARIMA model loaded from {path}")
