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


def main():
    config = load_config()

    exchange_name = config['exchange']
    symbol = config['symbol']
    timeframes = config['timeframes']
    future_periods = config['future_periods']

    data_fetcher = DataFetcher()

    divergence_data = {}
    training_data = pd.DataFrame()

    for timeframe in timeframes:
        data_name = f"BTC_USDT_{timeframe}.csv"
        if data_name in os.listdir(os.path.join(PROJECT_PATH, "data")):
            data_full_name = os.path.join(PROJECT_PATH, "data", data_name)
            df = pd.read_csv(data_full_name)
        else:
            df = data_fetcher.fetch_ohlcv(symbol, timeframe)

        if 'timeframe' not in df.columns:
            df['timeframe'] = timeframe

        df = DataPreprocess.calculate_technical_indicators(df)
        df = DataPreprocess.generate_features(df)
        # 미래 기간 설정
        future_period = future_periods[timeframe]
        df = DataPreprocess.generate_labels(df, future_periods=future_period)
        df = DataPreprocess.label_data(df, future_periods=future_period)
        df.dropna(inplace=True)
        df['timeframe'] = timeframe
        training_data = pd.concat([training_data, df])
        
        # 다이버전스 탐지 및 시각화
        divergence_df = DivergenceDetector.find_divergences(df)
        divergence_data[timeframe] = divergence_df
        # Visualizer.plot_divergence(df, divergence_df, title=f'BTC_USDT_{timeframe}', save_fig=True)
    
    training_data.reset_index(inplace=True)
    training_data.to_csv(f'{PROJECT_PATH}/data/training_data.csv', index=False)
    logging.info('Training data saved to data/training_data.csv')


if __name__ == "__main__":
    main()