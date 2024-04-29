import torch
from torch import nn
import torch.nn.functional as F
class ScaledDotProductAttention(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.scaling_factor = torch.rsqrt(torch.tensor(features, dtype=torch.float32))

    def forward(self, query, key, value):
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling_factor
        attention_weights = F.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_weights, value)

class AttentionModel(nn.Module):
    def __init__(self, input_dims=13,lstm_units=64,cnn_output=64,dropout_rate=0.3,kernel_size=1):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dims, cnn_output, kernel_size=kernel_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.bilstm = nn.LSTM(cnn_output, lstm_units, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.attention = ScaledDotProductAttention(lstm_units * 2)  # 2 for bidirection
        self.fc = nn.Linear(lstm_units * 2, 1)

    def forward(self, x):
        x = F.relu(self.conv1d(x))
        x = self.dropout1(x)
        x = x.transpose(1, 2)  # Add this line
        x, _ = self.bilstm(x)
        x = self.dropout2(x)
        x = self.attention(x, x, x)  # query, key, and value are all x
        x = torch.sigmoid(self.fc(x))
        return x

