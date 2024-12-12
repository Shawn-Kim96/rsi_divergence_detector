import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_size, num_classes, num_layers=2, num_heads=4, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(feature_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, num_classes)

    def forward(self, src):
        # src shape: [batch_size, seq_length, feature_size]
        src = src.permute(1, 0, 2)  # Transformer expects input shape: [seq_length, batch_size, feature_size]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)  # Global average pooling
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices in the array
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices in the array
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]  # [seq_length, batch_size, d_model]
        return self.dropout(x)
