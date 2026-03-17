import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero and handle NaN
    mask = (y_true != 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy as (1 - MAPE) * 100.
    Returns a value between 0 and 100.
    """
    mape = calculate_mape(y_true, y_pred)
    # Convert MAPE (percentage) to accuracy percentage
    accuracy = 100 - mape
    # Handle cases where MAPE > 100% or is invalid
    if np.isnan(accuracy) or accuracy < 0:
        return 0.0
    return min(accuracy, 100.0)

def direction_accuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Check if sign matches
    correct_direction = (np.sign(y_true) == np.sign(y_pred)).sum()
    return correct_direction / len(y_true)
