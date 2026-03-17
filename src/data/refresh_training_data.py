import sys
import os

# Add src to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_binance import fetch_ohlcv, save_to_parquet, load_config
from data.preprocess import preprocess_data
from features.indicators import add_indicators
from features.lag_features import add_lags_and_target

def run_refresh():
    print("=== Starting Full Data Refresh Pipeline ===")
    config = load_config()
    data_config = config['data']
    
    # 1. Fetch
    print("\n[1/4] Fetching fresh data...")
    df_raw = fetch_ohlcv(
        symbol=data_config['symbol'],
        timeframe=data_config['timeframe'],
        limit=data_config['limit']
    )
    
    # 2. Preprocess
    print("\n[2/4] Preprocessing...")
    df_clean = preprocess_data(df_raw)
    save_to_parquet(df_clean, data_config['raw_path'])
    
    # 3. Indicators
    print("\n[3/4] Adding indicators...")
    df_indicators = add_indicators(df_clean)
    
    # 4. Lags & Target
    print("\n[4/4] Finalizing features...")
    df_final = add_lags_and_target(df_indicators)
    
    # Drop NaNs and Save
    df_final = df_final.dropna()
    processed_path = data_config['processed_path']
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_final.to_parquet(processed_path, index=False)
    
    print(f"\nSUCCESS: Training data refreshed and saved to {processed_path}")
    print(f"Total samples: {len(df_final)}")
    print(f"Data range: {df_final['timestamp'].min()} to {df_final['timestamp'].max()}")

if __name__ == "__main__":
    run_refresh()
