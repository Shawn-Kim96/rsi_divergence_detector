import talib
import pandas as pd

class DataPreprocess:
    @staticmethod
    def calculate_rsi(df, period=14):
        df = df.copy()
        df['rsi'] = talib.RSI(df['close'], timeperiod=period)
        return df

    @staticmethod
    def calculate_technical_indicators(df):
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['willr'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        return df

    @staticmethod
    def generate_features(df):
        df = df.copy()
        # Lag features
        df['return_1'] = df['close'].pct_change()
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)
        # Rolling statistics
        df['volatility_5'] = df['return_1'].rolling(window=5).std()
        df['volatility_10'] = df['return_1'].rolling(window=10).std()
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_rolling_mean'] = df['volume'].rolling(window=5).mean()
        # 피처 엔지니어링 코드
        return df

    @staticmethod
    def generate_labels(df, future_periods):
        df = df.copy()
        for period in future_periods:
            df[f'future_return_{period}'] = df['close'].shift(-period) / df['close'] - 1
        return df

    @staticmethod
    def label_data(df, future_periods, timeframe=None):
        df = df.copy()
        timeframes = df['timeframe'].unique()
        df_list = []
        
        for timeframe in timeframes:
            df_timeframe = df[df['timeframe'] == timeframe]
            for period in future_periods:
                future_return_col = f'future_return_{period}'
                zscore_col = f'future_return_zscore_{period}'
                label_col = f'label_{period}'
                
                # future_return이 존재하는 경우에만 계산
                if future_return_col in df_timeframe.columns:
                    # Z-score 계산
                    df_timeframe[zscore_col] = (df_timeframe[future_return_col] - df_timeframe[future_return_col].mean()) / df_timeframe[future_return_col].std()
                    # 레이블링
                    df_timeframe[label_col] = 0  # Neutral
                    df_timeframe.loc[df_timeframe[zscore_col] > 0.5, label_col] = 1  # Buy
                    df_timeframe.loc[df_timeframe[zscore_col] < -0.5, label_col] = -1  # Sell
            df_list.append(df_timeframe)
        
        df_labeled = pd.concat(df_list)
        return df_labeled

