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

PROJECT_PATH = "/Users/shawn/Documents/personal/rsi_divergence_detector"
CUR_DIR = PROJECT_PATH + "/src/data_process"

load_dotenv(f"{PROJECT_PATH}/.env")
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


print(CUR_DIR)
def load_config(config_file=os.path.join(CUR_DIR, 'config.yaml')):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def fetch_data_from_external(exchange_name, symbol, timeframe):
    data_fetcher = DataFetcher(exchange_name=exchange_name)
    
    data_name = f"BTC_USDT_{timeframe}.csv"

    if data_name in os.listdir(os.path.join(PROJECT_PATH, "data")):
        data_full_name = os.path.join(PROJECT_PATH, "data", data_name)
        df = pd.read_csv(data_full_name)
    else:
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
    start_time = time.time()
    
    data_labeler = DataLabeler(price_data = training_data.loc[training_data.timeframe == '5m'])
    for key, value in divergence_data.items():
        divergence_data[key] = data_labeler.label_divergence_data(value)
    
    print(f"Finished labeling data. Processed time :: {time.time() - start_time}")
    pd.to_pickle(training_data, f'{PROJECT_PATH}/data/training_data.pickle')

    print('Training data saved and divergence data saved.')

    
def update_divergence_data():
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
    start_time = time.time()

    data_labeler = DataLabeler(price_data = training_data.loc[training_data.timeframe == '5m'])
    for key, value in divergence_data.items():
        divergence_data[key] = data_labeler.label_divergence_data(value)

    print(f"Finished labeling data. Processed time :: {time.time() - start_time}")


    pd.to_pickle(divergence_data, f"{PROJECT_PATH}/data/divergence_data_updated.pickle")



if __name__ == "__main__":
    main()