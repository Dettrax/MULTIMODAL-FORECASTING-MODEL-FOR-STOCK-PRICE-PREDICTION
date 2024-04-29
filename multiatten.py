import torch
from torch import nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, input_dims, lstm_units=64, cnn_output=64, dropout_rate=0.3, kernel_size=1, num_heads=8):
        super().__init__()
        lstm_units = int(lstm_units/num_heads) * num_heads

        self.bilstm = nn.LSTM(cnn_output, lstm_units, bidirectional=True, batch_first=True)
        self.multihead_attention = nn.MultiheadAttention(lstm_units * 2, num_heads)
        self.conv1d = nn.Conv1d(input_dims, cnn_output, kernel_size=kernel_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_units * 2, 1)

    def forward(self, x):

        x = F.relu(self.conv1d(x))
        x = self.dropout1(x)
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)
        x = self.dropout2(x)
        x = x.transpose(0, 1)
        x, _ = self.multihead_attention(x, x, x)
        x = x.transpose(0, 1)
        x = torch.mean(x, dim=1)
        x = torch.sigmoid(self.fc(x))
        return x

