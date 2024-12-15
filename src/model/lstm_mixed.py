import torch
import torch.nn as nn


# ----------------------------------------
# Model Definition
# ----------------------------------------
class MixedLSTMModel(nn.Module):
    def __init__(self, seq_input_dim, seq_hidden_dim=64, seq_num_layers=2,
                 nonseq_input_dim=0, mlp_hidden_dim=64, num_classes=2, dropout=0.1):
        super(MixedLSTMModel, self).__init__()
        self.seq_num_layers = seq_num_layers
        self.seq_hidden_dim = seq_hidden_dim

        # LSTM for sequential data
        self.lstm = nn.LSTM(
            input_size=seq_input_dim, 
            hidden_size=seq_hidden_dim,
            num_layers=seq_num_layers, 
            batch_first=True, 
            dropout=dropout, 
            bidirectional=False
        )

        # Fully-connected layers after LSTM
        fc_input_dim = seq_hidden_dim + nonseq_input_dim

        self.fc1 = nn.Linear(fc_input_dim, mlp_hidden_dim)
        self.relu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_linear = nn.Dropout(0.3)
        self.fc2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim//2)
        self.fc3 = nn.Linear(mlp_hidden_dim//2, mlp_hidden_dim//2)
        self.fc4 = nn.Linear(mlp_hidden_dim//2, num_classes)

    def forward(self, seq_input, nonseq_input):
        """
        Forward pass of the model.
        
        Parameters:
        - seq_input: Tensor of shape (batch_size, seq_length, seq_input_dim)
        - nonseq_input: Tensor of shape (batch_size, nonseq_input_dim)
        
        Returns:
        - out: Tensor of shape (batch_size, num_classes)
        """
        # LSTM output
        lstm_out, (h_n, c_n) = self.lstm(seq_input)
        # h_n: (num_layers, batch_size, hidden_dim)
        h_last = h_n[-1]  # (batch_size, hidden_dim)

        if nonseq_input.shape[1] > 0:
            x = torch.cat((h_last, nonseq_input), dim=1)  # (batch_size, hidden_dim + nonseq_input_dim)
        else:
            x = h_last

        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout_linear(self.relu(self.fc2(x)))
        x = self.dropout_linear(self.relu(self.fc3(x)))
        out = self.fc4(x)
        
        return out