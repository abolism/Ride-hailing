import torch
import torch.nn as nn

class LSTMForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers>1 else 0.0,
            bidirectional=bidirectional
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)  # out: (batch, seq_len, hidden*directions)
        # choose last timestep output
        last = out[:, -1, :]  # (batch, hidden*directions)
        y = self.fc(last).squeeze(-1)  # (batch,)
        return y