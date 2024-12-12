import pandas as pd
import numpy as np
import torch
from datetime import timedelta
from torch.utils.data import Dataset

PROJECT_PATH = "/Users/shawn/Documents/personal/rsi_divergence_detector"

# Load the training data
df = pd.read_pickle(f'{PROJECT_PATH}/data/training_data.pickle')
divergence_df = pd.read_pickle(f"{PROJECT_PATH}/data/divergence_data2.pickle")


class LSTMDivergenceDataset(Dataset):
    def __init__(self, ddf, df5, seq_length=288):
        """
        ddf: divergence DataFrame with columns like ['end_datetime', 'label', 'TP_percent', 'SL_percent', 'div_5m', ...]
        df5: time series DataFrame (5-min intervals) with columns: ['open', 'high', 'low', 'close', 'volume', 'rsi', ...]
             Index is datetime or 'timestamp' column set as index.
        seq_length: number of timesteps in the sequence (288)
        
        We'll extract the time series ending at end_datetime + 15 minutes.
        """
        self.ddf = ddf
        self.df5 = df5
        self.seq_length = seq_length

        # Ensure df5 index is datetime and sorted
        if 'timestamp' in self.df5.columns:
            self.df5 = self.df5.set_index('timestamp')
        self.df5 = self.df5.sort_index()

        # We'll store column names
        self.ts_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'ema_12', 
                        'ema_26', 'bb_upper', 'bb_middle', 'bb_lower', 'adx', 'willr', 'cci', 'atr', 'return_1', 
                        'return_5', 'return_10', 'volatility_5', 'volatility_10', 'volume_change', 'volume_rolling_mean']
        
        # Non-sequential features from ddf (excluding label and times)
        self.nonseq_cols = ['TP_percent', 'SL_percent', 'TP', 'SL', 'divergence', 'div_5m', 'div_15m', 'div_1h', 'div_4h', 'div_1d', 'TP_/_SL']
        # Filter only the columns that actually exist in ddf
        self.nonseq_cols = [c for c in self.nonseq_cols if c in ddf.columns]

        # Convert labels to int if they are bool
        if self.ddf['label'].dtype == bool:
            self.ddf['label'] = self.ddf['label'].astype(int)

        self.events = self.ddf.index  # This is start_datetime as the index of ddf

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        start_dt = self.events[idx]
        row = self.ddf.loc[start_dt]

        end_dt = row['end_datetime']
        # last time step = end_dt + 15 min
        last_time = pd.to_datetime(end_dt) + timedelta(minutes=25)
        # We want seq_length steps ending at last_time with frequency 5min
        # This implies start_time = last_time - (seq_length-1)*5min
        start_time = last_time - timedelta(minutes=(self.seq_length-1)*5)

        # Extract time series slice
        ts_slice = self.df5.loc[start_time:last_time]

        # If we don't have enough data points due to missing data, we may need to pad
        if len(ts_slice) < self.seq_length:
            # Pad at the start
            needed = self.seq_length - len(ts_slice)
            # Create a dummy df with NaN and ffill later
            pad_df = pd.DataFrame(index=pd.date_range(start=start_time, periods=self.seq_length, freq='5T'), columns=self.df5.columns)
            pad_df.loc[ts_slice.index, :] = ts_slice
            pad_df = pad_df.ffill().bfill() # fill missing values if needed
            ts_slice = pad_df

        # Ensure exact length
        ts_slice = ts_slice.iloc[-self.seq_length:]

        # Extract features
        X_seq = ts_slice[self.ts_cols].values  # shape: (seq_length, num_features)
        X_nonseq = row[self.nonseq_cols].values.astype(float) if len(self.nonseq_cols)>0 else np.array([])

        y = row['label']

        return (torch.tensor(X_seq, dtype=torch.float32),
                torch.tensor(X_nonseq, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long))