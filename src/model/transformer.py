import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math


class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    """
    def __init__(self, d_model, max_len=5000):
        """
        Initializes the positional encoding.
        
        Parameters:
        - d_model: Dimension of the embeddings.
        - max_len: Maximum length of the sequences.
        """
        super(PositionalEncoding, self).__init__()
        
        # Create a long enough P matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Adds positional encoding to input embeddings.
        
        Parameters:
        - x: Tensor of shape (batch_size, seq_length, d_model)
        
        Returns:
        - Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerMixedModel(nn.Module):
    """
    A Transformer-based model that processes sequential data and integrates non-sequential features.
    """
    def __init__(self, seq_input_dim, seq_model_dim=128, seq_num_heads=8, seq_num_layers=3,
                 nonseq_input_dim=0, mlp_hidden_dim=256, num_classes=2, dropout=0.3, max_len=5000):
        """
        Initializes the TransformerMixedModel.
        
        Parameters:
        - seq_input_dim: Number of features in the sequential input.
        - seq_model_dim: Dimension of the Transformer embeddings.
        - seq_num_heads: Number of attention heads in the Transformer.
        - seq_num_layers: Number of Transformer encoder layers.
        - nonseq_input_dim: Number of non-sequential input features.
        - mlp_hidden_dim: Number of hidden units in the MLP layers.
        - num_classes: Number of output classes.
        - dropout: Dropout rate.
        - max_len: Maximum sequence length for positional encoding.
        """
        super(TransformerMixedModel, self).__init__()
        self.seq_model_dim = seq_model_dim
        self.layer_norm = nn.LayerNorm(seq_model_dim)
        self.input_proj = nn.Linear(seq_input_dim, seq_model_dim)
        self.pos_encoder = PositionalEncoding(d_model=seq_model_dim, max_len=max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=seq_model_dim, nhead=seq_num_heads, 
                                                    dim_feedforward=seq_model_dim*4, dropout=dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=seq_num_layers)
        self.dropout = nn.Dropout(dropout)
        
        # Fully Connected Layers
        fc_input_dim = seq_model_dim + nonseq_input_dim
        self.fc1 = nn.Linear(fc_input_dim, mlp_hidden_dim)
        self.relu = nn.LeakyReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_hidden_dim, num_classes)
        self.fc_residual = nn.Linear(fc_input_dim, mlp_hidden_dim)  # Residual path
    
        
    def forward(self, seq_input, nonseq_input):
        """
        Forward pass of the model.
        
        Parameters:
        - seq_input: Tensor of shape (batch_size, seq_length, seq_input_dim)
        - nonseq_input: Tensor of shape (batch_size, nonseq_input_dim)
        
        Returns:
        - out: Tensor of shape (batch_size, num_classes)
        """
        x = self.input_proj(seq_input)  # Shape: (batch_size, seq_length, seq_model_dim)
        x = self.pos_encoder(x)  # Shape: (batch_size, seq_length, seq_model_dim)
        x = self.layer_norm(x)
        x = x.transpose(0, 1)  # Prepare for Transformer: (seq_length, batch_size, seq_model_dim)
        x = self.transformer_encoder(x)  # Shape: (seq_length, batch_size, seq_model_dim)
        x = x.mean(dim=0)  # Shape: (batch_size, seq_model_dim)
        x = self.dropout(x)
        
        # Concatenate with non-sequential features if they exist
        if nonseq_input.size(1) > 0:
            x = torch.cat((x, nonseq_input), dim=1)  # Shape: (batch_size, seq_model_dim + nonseq_input_dim)
        
        # Fully Connected Layers with Residual Connection
        residual = self.fc_residual(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc(x)
        x = x + residual  # Residual connection
        out = self.fc2(x)
        
        return out
