import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Load the training data
# Replace 'training_data.csv' with your actual data file or DataFrame
df = pd.read_csv('training_data.csv', parse_dates=['timestamp'])

# Exclude 1-minute timeframe data
df = df[df['timeframe'] != '1m']

# Sort the data by timestamp and timeframe
df.sort_values(['timeframe', 'timestamp'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Load the divergence data
divergence_df = pd.read_csv('divergence_data.csv', parse_dates=['start_datetime', 'entry_datetime', 'previous_peak_datetime'])

# Merge divergence data with training data
# We'll use 'entry_datetime' to align divergence signals with the training data
df = df.merge(divergence_df[['entry_datetime', 'divergence', 'TP', 'SL']], how='left', left_on='timestamp', right_on='entry_datetime')

# Fill NaN divergence entries with 'No Divergence'
df['divergence'].fillna('No Divergence', inplace=True)

# Drop columns that won't be used
unused_columns = ['timestamp', 'timeframe', 'entry_datetime', 'start_datetime', 'previous_peak_datetime']
df.drop(columns=unused_columns, inplace=True, errors='ignore')

# Handle categorical variables if any (e.g., 'divergence')
df['divergence_label'] = df['divergence'].map({'Bullish Divergence': 1, 'Bearish Divergence': -1, 'No Divergence': 0})

# Drop the original 'divergence' column
df.drop(columns=['divergence'], inplace=True)

# Identify feature columns and label columns
feature_cols = [col for col in df.columns if not col.startswith('future_return') and not col.startswith('label_') and col != 'divergence_label']
label_cols = ['divergence_label']  # You can include 'future_return' columns if needed

# Fill missing values
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
df.dropna(inplace=True)

# Scale numerical features
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])


sequence_length = 60  # You can adjust this

# Group data by timeframe if necessary
grouped = df.groupby('timeframe')

sequences = []
labels = []

for _, group in grouped:
    data = group[feature_cols + label_cols].values
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        sequences.append(seq[:, :-1])  # Features
        labels.append(seq[-1, -1])     # Label at the end of the sequence

# Convert to numpy arrays
sequences = np.array(sequences)
labels = np.array(labels)
