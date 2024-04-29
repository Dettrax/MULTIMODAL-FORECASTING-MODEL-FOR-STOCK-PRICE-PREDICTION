import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.attention_weights = nn.Parameter(torch.Tensor(1, features))
        nn.init.xavier_uniform_(self.attention_weights)  # Add this line

    def forward(self, x):
        weights = F.softmax(F.relu(torch.matmul(x, self.attention_weights.t())), dim=-1)
        return torch.sum(weights * x, dim=1)
class AttentionModel(nn.Module):
    def __init__(self, input_dims=13,lstm_units=64,cnn_output=64,dropout_rate=0.3,kernel_size=1):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dims, cnn_output, kernel_size=kernel_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.bilstm = nn.LSTM(cnn_output, lstm_units, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.attention = Attention(lstm_units * 2)  # 2 for bidirection
        self.fc = nn.Linear(lstm_units * 2, 1)

    def forward(self, x):
        x = F.relu(self.conv1d(x))
        x = self.dropout1(x)
        x = x.transpose(1, 2)  # Add this line
        x, _ = self.bilstm(x)
        x = self.dropout2(x)
        x = self.attention(x)
        x = torch.sigmoid(self.fc(x))
        return x

# # Instantiate the model
# model = AttentionModel(input_dims=13, time_steps=20, lstm_units=64)
#
# # Create a random input tensor
# input_tensor = torch.randn(32, 13, 10)  # batch_size=32, sequence_length=10, input_dims=13
#
# # Make a forward pass through the model
# output = model(input_tensor)
