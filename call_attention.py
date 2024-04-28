import pandas as pd
import numpy as np
from attention_model import *
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from tqdm import tqdm
import torch.nn as nn
from sklearn import metrics
np.random.seed(10)
torch.manual_seed(10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('brent_processed.csv')
data = data.dropna(axis=0, how='any')
data.index = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data =data.drop(['Date'], axis=1)
data = pd.DataFrame(data, dtype=np.float64)

data.sort_index(inplace=True)

test_split = '2023-09-20' #150
test_beg = '2023-09-21'

residuals = pd.read_csv('./ARIMA_residuals1.csv')
residuals.index = pd.to_datetime(residuals['Date'])
residuals.pop('Date')
merge_data = pd.merge(data, residuals, on='Date')

merge_idx = merge_data.index
merge_cols = merge_data.columns
#apply minmaxscaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
merge_data = scaler.fit_transform(merge_data)
merge_data = pd.DataFrame(merge_data, index=merge_idx, columns=merge_cols)

train = merge_data.loc[:test_split]
test = merge_data.loc[test_beg:]

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)


def create_sequences(data, lookback):
    x = []
    y = []
    for i in range(lookback, data.shape[0]):
        x.append(data[i - lookback: i])
        y.append(data[i, 0])
    return torch.tensor(x, device=device), torch.tensor(y, device=device)


def optimizer_att(train, test,cnn_output, hidden_dim, dropout_rate,lookback=10, num_epochs=100, batch_size=64, lr=0.001, weight_decay=0.001):
    model = AttentionModel(input_dims=len(train.columns), lstm_units=hidden_dim,cnn_output=cnn_output,dropout_rate=dropout_rate).to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    # Create sequences for LSTM
    lookback = lookback
    trainX, trainY = create_sequences(train.values, lookback)
    testX, testY = create_sequences(test.values, lookback)
    # Train the model
    num_epochs = num_epochs
    batch_size = batch_size
    model.train()
    # Training loop for the best model
    for epoch in range(num_epochs):
        for i in range(0, len(trainX), batch_size):
            inputs = trainX[i:i + batch_size]
            targets = trainY[i:i + batch_size]
            inputs = inputs.detach().cpu()
            inputs = inputs.permute(0, 2, 1)
            inputs = torch.tensor(inputs.float(), device=device)
            print(f"inputs type before forward pass: {inputs.dtype}")  # Add this line

            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs.squeeze(), targets.float())
            loss.backward()
            optimizer.step()
        # if epoch % 10 == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         outputs = model(testX.float())
        #         val_loss = criterion(outputs.squeeze(), testY.float())
        #     print(f'Epoch {epoch + 1}, Loss: {val_loss.item():.4f}')


    def calculate_metrics(y_true, y_pred):
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()

        mse = metrics.mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)

        return mse, rmse, mae, r2

    # After training your model, you can calculate the metrics
    model.eval()
    with torch.no_grad():
        testX = testX.permute(0, 2, 1)
        print(f"testX type before forward pass: {testX.dtype}")  # Add this line
        outputs = model(testX.to(torch.float32).to(device))
        mse, rmse, mae, r2 = calculate_metrics(testY, outputs.squeeze())
        print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}')
    return r2


def objective(trial):
    # Define the hyperparameters to optimize
    # model_type = trial.suggest_categorical('model_type', [1, 2, 3])
    cnn_output = trial.suggest_int('cnn_output', 32, 128)
    hidden_dim = trial.suggest_int('hidden_dim', 10, 100)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    lookback = trial.suggest_int('lookback', 1, 20)
    num_epochs = trial.suggest_int('num_epochs', 50, 300)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)

    # Optimize the hyperparameters
    r2 = optimizer_att(train, test,cnn_output, hidden_dim, dropout_rate,lookback, num_epochs, batch_size, lr, weight_decay)

    # Return the R2 score as the value to maximize
    return r2


# Create a study object and optimize the objective function
sampler = TPESampler(seed=10)
pruner = SuccessiveHalvingPruner()
study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=100,n_jobs=1)

# Print the best parameters
print(study.best_params)



