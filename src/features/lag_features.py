import pandas as pd
import numpy as np
import yaml
import os

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def add_lags_and_target(df):
    print("Calculating lag features and target variable...")
    config = load_config()
    feat_config = config['features']
    
    # Lag Features
    for lag in feat_config['lags']:
        df[f'lag_{lag}'] = df['close'].shift(lag)
    
    # Target Variable: next_return = log(price_t+1 / price_t)
    df['next_return'] = np.log(df['close'].shift(-1) / df['close'])
    
    return df

def main():
    config = load_config()
    raw_path = config['data']['raw_path']
    processed_path = config['data']['processed_path']
    
    try:
        # Read from processed_path if it exists (indicators might be there), otherwise raw
        source_path = processed_path if os.path.exists(processed_path) else raw_path
        df = pd.read_parquet(source_path)
        df_final = add_lags_and_target(df)
        
        # Drop rows with NaN from lags and target
        df_final = df_final.dropna()
        
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df_final.to_parquet(processed_path, index=False)
        print(f"Feature engineering complete. File saved to {processed_path}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {raw_path}.")

if __name__ == "__main__":
    main()
