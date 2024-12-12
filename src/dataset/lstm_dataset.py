import os
import logging
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

PROJECT_PATH = "/Users/shawn/Documents/personal/rsi_divergence_detector"

# ----------------------------------------
# Logging configuration
# ----------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------
# Example Data Preparation
# ----------------------------------------
def create_divergence_sequence(price_df, divergence_data):
    """
    Create a divergence feature sequence aligned with price_df timestamps for multiple timeframes.
    divergence_data: { '5m': divergence_df_5m, '1h': divergence_df_1h, ... }
    Each divergence_df_x: has divergence info with timestamps. For each timestamp in price_df, 
    we check if there's a bullish divergence (1), bearish divergence (-1), else 0.
    
    This is just an example. You need to adapt it based on how you store divergence info.
    """
    # Initialize a dict of arrays, one per timeframe
    divergence_arrays = {}
    for tf, divergence_df in divergence_data.items():
        # Assume divergence_df indexed by datetime with a column 'divergence' that can be 'Bullish' or 'Bearish' or None
        # We'll create an array aligned with price_df index
        arr = np.zeros(len(price_df), dtype=np.float32)
        # For each event in divergence_df, mark the corresponding timestamps
        # This is a simplistic approach: if end_datetime in divergence_df matches a price_df timestamp,
        # set arr[idx] = 1 for bullish, -1 for bearish.
        # You may need a more sophisticated approach depending on your data.
        for start_dt, event in divergence_df.iterrows():
            end_dt = event['end_datetime']
            if start_dt in price_df.index and end_dt in price_df.index:
                for time_delta in range(0, int((end_dt - start_dt).total_seconds()) + 1, 5*60):
                    t = start_dt + pd.Timedelta(seconds=time_delta)
                    if t in price_df.index:
                        idx = price_df.index.get_loc(t)
                        if event['divergence'] == 'Bullish Divergence':
                            arr[idx] = 1
                        elif event['divergence'] == 'Bearish Divergence':
                            arr[idx] = -1
        divergence_arrays[tf] = arr
    
    # Combine into a single DataFrame (or array)
    # For simplicity, just stack them column-wise
    for tf, arr in divergence_arrays.items():
        price_df[f"div_{tf}"] = arr
    
    return price_df


class LSTMDivergenceDataset(Dataset):
    def __init__(self, divergence_df, price_df, divergence_data,
                 seq_length=288, 
                 scaler=None,
                 ts_cols=None,
                 nonseq_cols=None):
        """
        divergence_df: DataFrame of divergence events. Indexed by start_datetime (unique events).
             Must contain 'end_datetime', 'label', and other columns.
        price_df: 5-min price/indicator DataFrame indexed by datetime. Contains at least the columns in ts_cols.
        divergence_data: dict of { timeframe: divergence_df_timeframe }, to create divergence features sequences.
        
        seq_length: number of timesteps in each sequence.
        scaler: a fitted scaler for sequential data normalization, if None we fit internally.
        ts_cols: columns in price_df used as sequential features.
        nonseq_cols: columns in divergence_df used as non-sequential features.
        """
        self.divergence_df = divergence_df.copy()
        self.price_df = price_df.copy()
        self.seq_length = seq_length

        self.not_ts_cols = ['timestamp', 'timeframe', 'volume_change']
        # Default columns if not provided
        if ts_cols is None:
            self.ts_cols = [x for x in self.price_df.columns if x not in self.not_ts_cols]
        else:
            self.ts_cols = ts_cols

        if nonseq_cols is None:
            self.nonseq_cols = ['TP_percent', 'SL_percent', 'TP', 'SL', 'divergence', 'TP_/_SL']
            self.nonseq_cols = [c for c in self.nonseq_cols if c in self.divergence_df.columns]
        else:
            self.nonseq_cols = nonseq_cols

        # Create additional divergence feature sequences from multiple timeframes
        self.divergence_seq = create_divergence_sequence(self.price_df, divergence_data)
        # Merge these divergence columns with price_df sequential data
        # This adds columns like 'div_5m', 'div_1h' etc. as sequential features
        # for c in self.divergence_seq.columns:
        #     self.price_df[c] = self.divergence_seq[c]
        #     if c not in self.ts_cols:
        #         self.ts_cols.append(c)

        # Ensure price_df is sorted
        self.price_df = self.price_df.sort_index()

        # Convert label to int if bool
        if self.divergence_df['label'].dtype == bool:
            self.divergence_df['label'] = self.divergence_df['label'].astype(int)

        # Fit scaler if needed
        if scaler is None:
            self.scaler = StandardScaler()
            # Fit on the entire price_df for simplicity
            self.scaler.fit(self.price_df[self.ts_cols].values)
        else:
            self.scaler = scaler

        self.events = self.divergence_df.index  # start_datetime
        

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        start_dt = self.events[idx]
        row = self.divergence_df.loc[start_dt]

        end_dt = row['end_datetime']
        # last time step = end_dt + 15min
        last_time = pd.to_datetime(end_dt) + timedelta(minutes=15)
        start_time = last_time - timedelta(minutes=(self.seq_length-1)*5)

        # Extract time series slice
        ts_slice = self.price_df.loc[start_time:last_time]

        # If we don't have enough data, pad
        if len(ts_slice) < self.seq_length:
            needed = self.seq_length - len(ts_slice)
            pad_index = pd.date_range(end=last_time, periods=self.seq_length, freq='5T')
            pad_df = pd.DataFrame(index=pad_index, columns=self.price_df.columns)
            pad_df.loc[ts_slice.index, :] = ts_slice
            pad_df = pad_df.ffill().bfill()
            ts_slice = pad_df

        ts_slice = ts_slice.iloc[-self.seq_length:]

        # Extract and normalize sequential features
        X_seq = ts_slice[self.ts_cols].values
        X_seq = self.scaler.transform(X_seq)

        # Non-sequential features
        X_nonseq = row[self.nonseq_cols].values.astype(float) if len(self.nonseq_cols) > 0 else np.array([])

        y = int(row['label'])

        return (torch.tensor(X_seq, dtype=torch.float32),
                torch.tensor(X_nonseq, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long))
