import numpy as np
import pandas as pd
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

class LSTMModel:
    """
    LSTM model implementation for BTC price prediction.
    """
    def __init__(self, input_shape=(24, 1), units=50):
        self.input_shape = input_shape
        self.units = units
        self.model = self._build_model()
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def _build_model(self):
        model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=self.input_shape),
            Dropout(0.2),
            LSTM(self.units, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def prepare_data(self, data, lookback=24):
        """
        Prepares sequences for LSTM training.
        """
        scaled_data = self.scaler_x.fit_transform(data)
        self.scaler_y.fit(data[:, [0]])
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0]) # Assuming target is the first column
        return np.array(X), np.array(y)

    def train(self, X, y, epochs=10, batch_size=32):
        print("Training LSTM model...")
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path + ".h5")
        joblib.dump(self.scaler_x, path + "_scaler_x.pkl")
        joblib.dump(self.scaler_y, path + "_scaler_y.pkl")
        print(f"LSTM model and scalers saved to {path}")

    def load_model(self, path: str):
        if not os.path.exists(path + ".h5"):
            # Try without .h5 if it's a directory
            if not os.path.exists(path):
                raise FileNotFoundError(f"No model found at {path}")
            self.model = tf.keras.models.load_model(path)
        else:
            self.model = tf.keras.models.load_model(path + ".h5")
        
        self.scaler_x = joblib.load(path + "_scaler_x.pkl")
        self.scaler_y = joblib.load(path + "_scaler_y.pkl")
        print(f"LSTM model and scalers loaded from {path}")
