import ccxt
import pandas as pd
import yaml
import os

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def fetch_ohlcv(symbol, timeframe, limit):
    print(f"Fetching {symbol} {timeframe} data from Binance...")
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    # Localize to UTC and convert to IST
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    return df

def save_to_parquet(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Data saved to {path}")

if __name__ == "__main__":
    config = load_config()
    data_config = config['data']
    
    df = fetch_ohlcv(
        symbol=data_config['symbol'],
        timeframe=data_config['timeframe'],
        limit=data_config['limit']
    )
    
    save_to_parquet(df, data_config['raw_path'])
