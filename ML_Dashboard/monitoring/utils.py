import pandas as pd
import numpy as np

def calculate_metrics(df):
    metrics = {}
    models = {
        'arima_price': 'ARIMA',
        'lr_price': 'Linear Regression',
        'lstm_price': 'LSTM',
        'rnn_price': 'RNN'
    }
    
    # Ensure dataframe copy and calculate consistent normalization reference
    df_clean = df.copy()
    mean_actual_price = df_clean['actual_price'].mean() if not df_clean['actual_price'].empty else 0
    
    for col, label in models.items():
        if col not in df_clean.columns:
            metrics[label] = {'mae': 0, 'rmse': float('inf'), 'mape': 100, 'accuracy': 0, 'nmae': 0, 'nrmse': 0}
            continue
            
        # Nullify any invalid values
        df_clean.loc[df_clean[col] < 1000, col] = np.nan
        
        valid_df = df_clean.dropna(subset=['actual_price', col])
        
        if len(valid_df) > 0 and mean_actual_price > 0:
            actual = valid_df['actual_price']
            pred = valid_df[col]
            
            mae = (actual - pred).abs().mean()
            rmse = np.sqrt(((actual - pred)**2).mean())
            
            # Accuracy as 1 - MAPE
            mape_value = ((actual - pred).abs() / actual).mean()
            accuracy = max(0, (1 - mape_value) * 100)
            
            # Normalized metrics (0-1 range)
            nmae = mae / mean_actual_price
            nrmse = rmse / mean_actual_price
            
            metrics[label] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape_value * 100,
                'accuracy': accuracy,
                'nmae': nmae,
                'nrmse': nrmse
            }
        else:
            metrics[label] = {'mae': 0, 'rmse': float('inf'), 'mape': 100, 'accuracy': 0, 'nmae': 1.0, 'nrmse': 1.0}
            
    return metrics
