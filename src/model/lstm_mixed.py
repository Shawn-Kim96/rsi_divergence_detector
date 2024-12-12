import torch
import torch.nn as nn


class MixedModel(nn.Module):
    def __init__(self, seq_input_dim, seq_hidden_dim, seq_num_layers, nonseq_input_dim, num_classes=2):
        super(MixedModel, self).__init__()
        
        # LSTM for sequential data
        self.lstm = nn.LSTM(input_size=seq_input_dim, hidden_size=seq_hidden_dim, num_layers=seq_num_layers, batch_first=True)
        
        # After LSTM, we get hidden state which we'll combine with non-seq features
        combined_input_dim = seq_hidden_dim + nonseq_input_dim
        
        self.fc = nn.Linear(combined_input_dim, num_classes)
        
    def forward(self, seq_input, nonseq_input):
        # seq_input: (B, L, D)
        # nonseq_input: (B, N)
        _, (h_n, _) = self.lstm(seq_input)  # h_n: (num_layers, B, seq_hidden_dim)
        # Take the last layer hidden state
        h_last = h_n[-1]  # (B, seq_hidden_dim)
        
        # Concatenate with nonseq input
        if nonseq_input.shape[1] > 0:
            x = torch.cat([h_last, nonseq_input], dim=1)  # (B, seq_hidden_dim + nonseq_input_dim)
        else:
            x = h_last
        
        out = self.fc(x)  # (B, num_classes)
        return out
