import torch
import torch.nn as nn


class CryptoLSTM(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, dropout=0.2):
        super(CryptoLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Red LSTM más profunda para leer indicadores complejos
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Capas densas de decisión
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)  # Predicción final del log_return

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Tomamos solo el resultado de la última vela de la secuencia
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out