import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression

class LinearRegressionModel:
    """
    Linear Regression model implementation using scikit-learn.
    """
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the Linear Regression model.
        """
        print("Training Linear Regression model...")
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):
        """
        Makes predictions using the trained model.
        """
        if self.model is None:
            raise Exception("Model must be trained before prediction.")
        return self.model.predict(X)

    def save_model(self, path: str):
        """
        Saves the trained model to a file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Linear Regression model saved to {path}")

    def load_model(self, path: str):
        """
        Loads a trained model from a file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        self.model = joblib.load(path)
        print(f"Linear Regression model loaded from {path}")
