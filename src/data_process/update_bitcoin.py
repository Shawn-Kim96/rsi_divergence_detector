import os
import sys
from datetime import datetime, timedelta
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
    symbol = 'BTC/USDT'
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    output_dir = os.path.join(PROJECT_PATH, 'data')
    
    # Create data directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data fetcher
    data_fetcher = DataFetcher(exchange_name='binance')
    
    # Get current date for logging
    current_date = datetime.now().strftime('%Y-%m-%d')
    print(f"Starting Bitcoin data update on {current_date}")
    
    # Update data for each timeframe
    for timeframe in timeframes:
        file_path = os.path.join(output_dir, f'{symbol.replace("/", "_")}_{timeframe}.csv')
        
        try:
            # Check if file exists to determine if this is an update or initial download
            if os.path.exists(file_path):
                print(f"Updating existing {timeframe} data...")
                # Load existing data
                existing_data = data_fetcher.load_data(symbol, timeframe, input_dir=output_dir)
                
                # Get the last timestamp and add a small offset to avoid duplicates
                last_timestamp = existing_data.index[-1]
                since_date = (last_timestamp + timedelta(minutes=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
                
                # Fetch new data since the last timestamp
                new_data = data_fetcher.fetch_ohlcv(symbol, timeframe, since=since_date)
                
                if not new_data.empty:
                    # Combine existing and new data
                    combined_data = existing_data.append(new_data)
                    # Remove duplicates if any
                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                    # Save updated data
                    combined_data.to_csv(file_path)
                    print(f"Added {len(new_data)} new rows to {timeframe} data")
                else:
                    print(f"No new {timeframe} data available")
            else:
                print(f"No existing data found. Downloading {timeframe} data from scratch...")
                # Fetch data from the beginning
                df = data_fetcher.fetch_ohlcv(symbol, timeframe, since='2023-01-01T00:00:00Z')
                df.to_csv(file_path)
                print(f"Successfully saved {len(df)} rows of {timeframe} data to {file_path}")
        except Exception as e:
            print(f"Error processing {timeframe} data: {str(e)}")
    
    print("Bitcoin data update completed")

if __name__ == "__main__":
    main()