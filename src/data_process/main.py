from data_fetcher import DataFetcher
from data_preprocess import DataPreprocess
from divergence import DivergenceDetector
from visualization import Visualizer
from data_labeler import DataLabeler
import pandas as pd
import yaml
import logging
import os, sys
from dotenv import load_dotenv
from tqdm import tqdm
import time

path_splited = os.path.abspath('.').split('rsi_divergence_detector')[0]
PROJECT_PATH = os.path.join(path_splited, 'rsi_divergence_detector')
sys.path.append(PROJECT_PATH)


load_dotenv(f"{PROJECT_PATH}/.env")
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


# TODO: divergence function is not optimal. There are many things to update for optimal algorithms
#       2. Implement sliding window algorithm for real-time divergence detector
#       3. Implement SQL server database instead of Google Sheets


def load_config(config_file=os.path.join(PROJECT_PATH, 'config.yaml')):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def fetch_data_from_external(exchange_name, symbol, timeframe):
    data_fetcher = DataFetcher(exchange_name=exchange_name)
    
    data_name = f"BTC_USDT_{timeframe}.csv"
    raw_data_path = os.path.join(PROJECT_PATH, "data", "raw_data", data_name)

    # First check if data exists in raw_data folder
    if os.path.exists(raw_data_path):
        print(f"Loading data from raw_data folder: {raw_data_path}")
        df = pd.read_csv(raw_data_path)
    # Then check if data exists in data folder
    elif data_name in os.listdir(os.path.join(PROJECT_PATH, "data")):
        data_full_name = os.path.join(PROJECT_PATH, "data", data_name)
        print(f"Loading data from data folder: {data_full_name}")
        df = pd.read_csv(data_full_name)
    else:
        print(f"No local data found for {timeframe}, fetching from exchange")
        df = data_fetcher.fetch_ohlcv(symbol, timeframe)
    
    df['timeframe'] = timeframe
    return df


def preprocess_data(df, timeframe, future_period_list):
    data_preprocessor = DataPreprocess(sequence_length=30)

    # feature generation for data
    df = data_preprocessor.calculate_technical_indicators(df)
    df = data_preprocessor.generate_features(df)
    
    # label generation for data
    # df = data_preprocessor.generate_labels(df, future_periods=future_period_list)
    # df = data_preprocessor.label_data(df, future_periods=future_period_list)
    df.dropna(inplace=True)

    df = data_preprocessor.change_index_to_datetime(df)

    return df


def main():
    config = load_config()

    exchange_name = config['exchange']
    symbol = config['symbol']
    timeframes = config['timeframes']
    # future_period_hash = config['future_periods']
    future_periods = list(range(1, 61))

    divergence_data = {}
    training_data = pd.DataFrame()
    divergence_detector = DivergenceDetector()

    for timeframe in tqdm(timeframes, desc=f"Processing data"):
        # fetch data from external link or use downloaded data
        print(f"Reading {timeframe} csv data")
        df = fetch_data_from_external(exchange_name, symbol, timeframe)

        print(f"Processing {timeframe} csv data")
        # feature generation for data
        df = preprocess_data(df, timeframe, future_periods)

        # concating data to total training data
        training_data = pd.concat([training_data, df])
        
        print(f"Making divergence data from {timeframe} data")
        # divergence period generation
        df = training_data.loc[training_data.timeframe == timeframe]
        divergence_data[timeframe] = divergence_detector.find_divergences(df, bullish_rsi_threshold=35, bearish_rsi_threshold=65)

    
    # label data
    print("Labeling divergence data")

    st = time.time()
    data_labeler = DataLabeler(price_data = training_data.loc[training_data.timeframe == '5m'])
    for key, value in tqdm(divergence_data.items(), desc="Processing timeframes"):
        print(f"Labeling {key} timeframe divergences ({len(value)} records)")
        divergence_data[key] = data_labeler.label_divergence_data(value)

    print(f"Finished labeling data. Processed time :: {time.time() - st}")

    
    # joining different timeframe divergences
    print("Joining different timeframe divergence data")
    st = time.time()
    divergence_data = divergence_detector.compare_with_different_timeframes(divergence_data=divergence_data)
    print(f"Finished joining data. Processed time :: {time.time() - st}")

    pd.to_pickle(divergence_data, f"{PROJECT_PATH}/data/divergence_data.pickle")
    pd.to_pickle(training_data, f'{PROJECT_PATH}/data/training_data.pickle')
    print('Training data saved and divergence data saved.')

    
def regenerate_divergence_data():
    config = load_config()
    timeframes = config['timeframes']

    divergence_data = {}
    training_data = pd.read_pickle(f"{PROJECT_PATH}/data/training_data.pickle")
    divergence_detector = DivergenceDetector()

    for timeframe in tqdm(timeframes, desc=f"Processing data"):
        print(f"Making divergence data from {timeframe} data")
        # divergence period generation
        df = training_data.loc[training_data.timeframe == timeframe]
        divergence_data[timeframe] = divergence_detector.find_divergences(df, bullish_rsi_threshold=35, bearish_rsi_threshold=65)


    # label data
    print("Labeling divergence data")

    st = time.time()
    data_labeler = DataLabeler(price_data = training_data.loc[training_data.timeframe == '5m'])
    for key, value in tqdm(divergence_data.items(), desc="Processing timeframes"):
        print(f"Labeling {key} timeframe divergences ({len(value)} records)")
        divergence_data[key] = data_labeler.label_divergence_data(value)

    print(f"Finished labeling data. Processed time :: {time.time() - st}")

    pd.to_pickle(divergence_data, f"{PROJECT_PATH}/data/divergence_data_updated.pickle")


def update_divergence_data():
    config = load_config()
    timeframes = config['timeframes']

    divergence_data = pd.read_pickle(f"{PROJECT_PATH}/data/divergence_data.pickle")
    divergence_detector = DivergenceDetector()

    # joining different timeframe divergences
    print("Joining different timeframe divergence data")
    st = time.time()
    divergence_data = divergence_detector.compare_with_different_timeframes(divergence_data=divergence_data)
    print(f"Finished joining data. Processed time :: {time.time() - st}")

    pd.to_pickle(divergence_data, f"{PROJECT_PATH}/data/divergence_data_updated.pickle")


if __name__ == "__main__":
    main()
    # update_divergence_data()