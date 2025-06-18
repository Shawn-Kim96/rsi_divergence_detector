import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from data_fetcher import DataFetcher

# Add project path to system path
path_splited = os.path.abspath('.').split('rsi_divergence_detector')[0]
PROJECT_PATH = os.path.join(path_splited, 'rsi_divergence_detector')
sys.path.append(PROJECT_PATH)

# Load environment variables
load_dotenv(f"{PROJECT_PATH}/.env")

def main():
    # Define parameters
    symbol = 'ETH/USDT'
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    output_dir = os.path.join(PROJECT_PATH, 'data')
    
    # Create data directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data fetcher
    data_fetcher = DataFetcher(exchange_name='binance')
    
    # Get current date for logging
    current_date = datetime.now().strftime('%Y-%m-%d')
    print(f"Starting Ethereum data download on {current_date}")
    
    # Fetch and save data for each timeframe
    for timeframe in timeframes:
        print(f"Downloading {symbol} {timeframe} data...")
        try:
            # Fetch data
            df = data_fetcher.fetch_ohlcv(symbol, timeframe, since='2023-01-01T00:00:00Z')
            
            # Save to CSV
            file_path = os.path.join(output_dir, f'{symbol.replace("/", "_")}_{timeframe}.csv')
            df.to_csv(file_path)
            print(f"Successfully saved {len(df)} rows of {timeframe} data to {file_path}")
        except Exception as e:
            print(f"Error fetching {timeframe} data: {str(e)}")
    
    print("Ethereum data download completed")

if __name__ == "__main__":
    main()