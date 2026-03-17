import pandas as pd
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def validate_data(df):
    print("Validating dataset...")
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"Warning: Found null values:\n{null_counts[null_counts > 0]}")
    else:
        print("No null values found.")

    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Warning: Found {duplicates} duplicate rows.")
    else:
        print("No duplicate rows found.")

    # Check for missing timestamps (assuming 1h timeframe)
    df = df.sort_values('timestamp')
    expected_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='1h')
    missing_timestamps = expected_range.difference(df['timestamp'])
    
    if len(missing_timestamps) > 0:
        print(f"Warning: Found {len(missing_timestamps)} missing timestamps.")
    else:
        print("No missing timestamps found.")

def main():
    config = load_config()
    raw_path = config['data']['raw_path']
    
    try:
        df = pd.read_parquet(raw_path)
        validate_data(df)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_path}. Run fetch_binance.py first.")

if __name__ == "__main__":
    main()
