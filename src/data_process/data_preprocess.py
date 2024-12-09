import talib
import pandas as pd
import numpy as np


class DataPreprocess:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        # self.scaler = StandardScaler()
        self.feature_columns = None

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
    def change_index_to_datetime(df):
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        if df.index.dtype == 'int64':
            df.set_index('datetime', inplace=True)
        
        return df

    def preprocess_data(self, data, future_periods):
        # Data Cleaning
        data = data.copy()
        data.drop_duplicates(subset=['datetime', 'timeframe'], inplace=True)
        data.sort_values(by='datetime', inplace=True)
        data.reset_index(drop=True, inplace=True)
        data.fillna(method='ffill', inplace=True)
        data.dropna(inplace=True)  # Drop remaining NaNs

        # Feature Selection
        exclude_columns = ['index', 'datetime', 'timestamp']
        exclude_columns += [col for col in data.columns if 'future_return_zscore' in col or 'label_' in col]

        self.feature_columns = [col for col in data.columns if col not in exclude_columns]
        X = data[self.feature_columns]

        # One-hot encode 'timeframe'
        X = pd.get_dummies(X, columns=['timeframe'], prefix='tf')

        # Target Variables: Future returns for specified periods
        target_columns = []
        for periods in future_periods.values():
            target_columns.extend([f'future_return_{period}' for period in periods])

        target_columns = list(set(target_columns))  # Remove duplicates

        # Ensure all target columns are present
        missing_cols = [col for col in target_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing target columns: {missing_cols}")

        y = data[target_columns]

        # Split Data
        split_index = int(len(X) * 0.8)
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        # Scaling
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Sequence Generation
        X_train_seq, y_train_seq = self.create_sequences_multi(X_train_scaled, y_train.values)
        X_test_seq, y_test_seq = self.create_sequences_multi(X_test_scaled, y_test.values)

        return X_train_seq, X_test_seq, y_train_seq, y_test_seq

    def create_sequences_multi(self, X, y):
        X_seq = []
        y_seq = []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)