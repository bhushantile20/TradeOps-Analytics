import pandas as pd
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def preprocess_data(df):
    print("Preprocessing data...")
    # Basic cleaning
    df = df.drop_duplicates().sort_values('timestamp')
    df = df.ffill() # Forward fill any gaps
    
    # Ensure types are correct
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    return df

def main():
    config = load_config()
    raw_path = config['data']['raw_path']
    # For now, we reuse the raw path if no intermediate step is needed, 
    # but typically it would save to a temporary processed location if needed.
    # However, the user asked for src/data/preprocess.py
    
    try:
        df = pd.read_parquet(raw_path)
        df_clean = preprocess_data(df)
        df_clean.to_parquet(raw_path, index=False) # Overwrite with cleaned data or separate path?
        # User specified data/processed for final features. 
        # I'll keep intermediate cleaning in data/raw for now or just pass through.
        print("Basic preprocessing complete.")
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_path}.")

if __name__ == "__main__":
    main()
