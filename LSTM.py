import torch
from torch import nn

class SingleLayerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SingleLayerLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.lstm2 = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out = self.fc1(out[:, -1, :])
        out = out.unsqueeze(1)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.fc2(out[:, -1, :])
        return out

class MultiLayerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MultiLayerLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.lstm3 = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.lstm4 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.fc1(out[:, -1, :])
        out = out.unsqueeze(1)
        out, _ = self.lstm3(out)
        out = self.dropout3(out)
        out, _ = self.lstm4(out)
        out = self.dropout4(out)
        out = self.fc2(out[:, -1, :])
        return out

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(BiLSTM, self).__init__()
        self.bidirectional_lstm1 = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim*2, output_dim)  # 2 for bidirection
        self.bidirectional_lstm2 = nn.LSTM(output_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim*2, output_dim)  # 2 for bidirection

    def forward(self, x):
        out, _ = self.bidirectional_lstm1(x)
        out = self.dropout1(out)
        out = self.fc1(out[:, -1, :])
        out = out.unsqueeze(1)
        out, _ = self.bidirectional_lstm2(out)
        out = self.dropout2(out)
        out = self.fc2(out[:, -1, :])
        return out

def lstm(model_type, input_dim, hidden_dim, output_dim, dropout_rate):
    if model_type == 1:
        # single-layer LSTM
        model = SingleLayerLSTM(input_dim, hidden_dim, output_dim, dropout_rate)
    elif model_type == 2:
        # multi-layer LSTM
        model = MultiLayerLSTM(input_dim, hidden_dim, output_dim, dropout_rate)
    elif model_type == 3:
        # BiLSTM
        model = BiLSTM(input_dim, hidden_dim, output_dim, dropout_rate)
    else:
        raise ValueError("Invalid model_type")
    return model