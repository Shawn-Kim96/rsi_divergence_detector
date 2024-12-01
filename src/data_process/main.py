from data_fetcher import DataFetcher
from data_preprocess import DataPreprocess
from divergence import DivergenceDetector
from visualization import Visualizer
import pandas as pd
import yaml
import logging
import os, sys
from dotenv import load_dotenv
# from ...utils.path_finder import get_project_path


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


def preprocess_data(df, timeframe, future_period_hash):
    data_preprocessor = DataPreprocess(sequence_length=30)

    # feature generation for data
    df = data_preprocessor.calculate_technical_indicators(df)
    df = data_preprocessor.generate_features(df)
    
    # label generation for data
    future_period_list = future_period_hash[timeframe]
    df = data_preprocessor.generate_labels(df, future_periods=future_period_list)
    df = data_preprocessor.label_data(df, future_periods=future_period_list)
    df.dropna(inplace=True)

    return df


def main():
    config = load_config()

    exchange_name = config['exchange']
    symbol = config['symbol']
    timeframes = config['timeframes']
    future_period_hash = config['future_periods']

    divergence_data = {}
    training_data = pd.DataFrame()
    # training_data = pd.read_csv(f"{PROJECT_PATH}/data/training_data.csv")
    
    for timeframe in timeframes:
        # fetch data from external link or use downloaded data
        df = fetch_data_from_external(exchange_name, symbol, timeframe)

        # feature / label generation for data
        df = preprocess_data(df, timeframe, future_period_hash)

        # concating data to total training data
        training_data = pd.concat([training_data, df])
        
        # divergence period generation
        divergence_data[timeframe] = DivergenceDetector.find_divergences(df)
    
    pd.to_pickle(divergence_data, f"{PROJECT_PATH}/data/divergence_data")
    training_data.reset_index(inplace=True)
    training_data.to_csv(f'{PROJECT_PATH}/data/training_data.csv', index=False)

    logging.info('Training data saved to data/training_data.csv')

    
    all_future_periods = [period for periods in future_periods.values() for period in periods]
    data = DataPreprocess.generate_labels(data, future_periods=all_future_periods)

    # Preprocess data
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = DataPreprocess.preprocess_data(data, future_periods)



if __name__ == "__main__":
    main()