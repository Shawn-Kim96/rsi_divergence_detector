import os
import pandas as pd
import sys

# Add project path to system path
path_splited = os.path.abspath('.').split('rsi_divergence_detector')[0]
PROJECT_PATH = os.path.join(path_splited, 'rsi_divergence_detector')
sys.path.append(PROJECT_PATH)

def merge_data():
    """
    Merge the latest data (BTC_USDT_*.csv in data/) with old data (BTC_USDT_*.csv in data/raw_data/)
    and save the merged data to data/raw_data/
    """
    print("Starting data merging process...")
    
    # Define timeframes
    timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
    
    for timeframe in timeframes:
        # Define file paths
        latest_file = os.path.join(PROJECT_PATH, 'data', f'BTC_USDT_{timeframe}.csv')
        old_file = os.path.join(PROJECT_PATH, 'data', 'raw_data', f'BTC_USDT_{timeframe}.csv')
        output_file = os.path.join(PROJECT_PATH, 'data', 'raw_data', f'BTC_USDT_{timeframe}.csv')
        
        # Check if both files exist
        latest_exists = os.path.exists(latest_file)
        old_exists = os.path.exists(old_file)
        
        if not latest_exists and not old_exists:
            print(f"No data files found for {timeframe} timeframe. Skipping.")
            continue
        
        if latest_exists and not old_exists:
            print(f"Only latest data exists for {timeframe}. Copying to raw_data folder.")
            # Ensure raw_data directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            # Read and save the latest data
            latest_df = pd.read_csv(latest_file)
            latest_df.to_csv(output_file, index=False)
            print(f"Saved {len(latest_df)} rows to {output_file}")
            continue
        
        if not latest_exists and old_exists:
            print(f"Only old data exists for {timeframe}. No merging needed.")
            continue
        
        # Both files exist, merge them
        print(f"Merging data for {timeframe} timeframe...")
        
        # Read data
        latest_df = pd.read_csv(latest_file)
        old_df = pd.read_csv(old_file)
        
        # Ensure datetime column is in the right format
        if 'datetime' in latest_df.columns and 'datetime' in old_df.columns:
            latest_df['datetime'] = pd.to_datetime(latest_df['datetime'])
            old_df['datetime'] = pd.to_datetime(old_df['datetime'])
        
        # Combine data
        combined_df = pd.concat([old_df, latest_df])
        
        # Remove duplicates based on datetime
        if 'datetime' in combined_df.columns:
            combined_df.drop_duplicates(subset=['datetime'], keep='last', inplace=True)
            combined_df.sort_values(by='datetime', inplace=True)
        
        # Save merged data
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Saved {len(combined_df)} rows to {output_file}")
    
    print("Data merging completed!")

if __name__ == "__main__":
    merge_data()