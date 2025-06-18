#!/bin/bash

# Set the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project directory: $PROJECT_DIR"

# Create data directory if it doesn't exist
mkdir -p "$PROJECT_DIR/data"

# Activate Python environment if using one
# Uncomment and modify the following line if you're using a virtual environment
# source /path/to/your/venv/bin/activate

echo "Starting data download process..."

# Create a simple Python script to download both BTC and ETH data
cat > "$PROJECT_DIR/temp_download_script.py" << 'EOF'
import os
import sys
from datetime import datetime

# Add project path to system path
path_splited = os.path.abspath('.').split('rsi_divergence_detector')[0]
PROJECT_PATH = os.path.join(path_splited, 'rsi_divergence_detector')
sys.path.append(PROJECT_PATH)

# Import the DataFetcher class
from src.data_process.data_fetcher import DataFetcher

def main():
    # Initialize data fetcher
    data_fetcher = DataFetcher(exchange_name='binance')
    
    # Define parameters
    symbols = ['BTC/USDT', 'ETH/USDT']
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    output_dir = os.path.join(PROJECT_PATH, 'data')
    since_date = '2023-01-01T00:00:00Z'  # Start from 2023
    
    # Create data directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download data for each symbol
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Downloading {symbol} data for all timeframes")
        print(f"{'='*50}")
        
        # Use the existing fetch_and_save_all_data method
        data_fetcher.fetch_and_save_all_data(
            symbol=symbol,
            timeframes=timeframes,
            output_dir=output_dir,
            since=since_date
        )
        
        print(f"Completed downloading {symbol} data")

if __name__ == "__main__":
    main()
EOF

# Run the Python script
echo "Running data download script..."
python "$PROJECT_DIR/temp_download_script.py"

# Clean up the temporary script
rm "$PROJECT_DIR/temp_download_script.py"

echo "Data download process completed!"

# Optional: Generate pickle files for analysis
echo "Preprocessing data for analysis..."
python "$PROJECT_DIR/src/data_process/main.py"

echo "All done! Data is ready for analysis."