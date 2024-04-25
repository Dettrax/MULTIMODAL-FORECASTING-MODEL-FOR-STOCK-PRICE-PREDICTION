import pandas as pd
import numpy as np
from LSTM import *
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


def optimizer_lstm(train, test,model_type, hidden_dim, dropout_rate,lookback=10, num_epochs=100, batch_size=64, lr=0.001, weight_decay=0.001):
    model = lstm(model_type, input_dim=len(train.columns), hidden_dim=hidden_dim, output_dim=1,dropout_rate=dropout_rate).to(device)

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
        outputs = model(testX.float())
        mse, rmse, mae, r2 = calculate_metrics(testY, outputs.squeeze())
        # print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}')
    return r2,outputs


def objective(trial):
    # Define the hyperparameters to optimize
    # model_type = trial.suggest_categorical('model_type', [1, 2, 3])
    hidden_dim = trial.suggest_int('hidden_dim', 10, 100)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    lookback = trial.suggest_int('lookback', 1, 20)
    num_epochs = trial.suggest_int('num_epochs', 50, 300)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)

    # Optimize the hyperparameters
    r2 = optimizer_lstm(train, test, 1, hidden_dim, dropout_rate, lookback, num_epochs, batch_size, lr,
                        weight_decay)

    # Return the R2 score as the value to maximize
    return r2


# # Create a study object and optimize the objective function
# sampler = TPESampler(seed=10)
# pruner = SuccessiveHalvingPruner()
# study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
# study.optimize(objective, n_trials=100,n_jobs=-1)
#
# # Print the best parameters
# print(study.best_params)

best_params = {'hidden_dim': 84, 'dropout_rate': 0.10924601759225532, 'lookback': 1, 'num_epochs': 299, 'batch_size': 52, 'lr': 6.599320933831208e-05, 'weight_decay': 0.0001443204850539564}


# r2,outputs = optimizer_lstm(train, test, 1, best_params['hidden_dim'], best_params['dropout_rate'], best_params['lookback'], best_params['num_epochs'], best_params['batch_size'], best_params['lr'], best_params['weight_decay'])

model = lstm(model_type=2,input_dim=len(train.columns), hidden_dim=best_params['hidden_dim'], output_dim=1,dropout_rate=best_params['dropout_rate']).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

# Create sequences for LSTM
lookback = best_params['lookback']
testX, testY = create_sequences(merge_data.values, lookback)

trainX, trainY = create_sequences(train.values, lookback)

# Train the model
num_epochs = best_params['num_epochs']
batch_size = best_params['batch_size']
model.train()
# Training loop for the best model
for epoch in tqdm(range(num_epochs)):
    for i in range(0, len(trainX), batch_size):
        inputs = trainX[i:i + batch_size]
        targets = trainY[i:i + batch_size]
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs.squeeze(), targets.float())
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    outputs = model(testX.float())

# Create a placeholder DataFrame
lstm_output_df = pd.DataFrame(index=merge_data.index, columns=['LSTM_Output'])
lstm_output_df = lstm_output_df.fillna(np.nan)

# Fill the last part of the DataFrame with the LSTM model's output
lstm_output_df.iloc[-len(outputs):] = outputs.cpu().numpy()

# Concatenate the LSTM output with the original data
merge_data_with_output = pd.concat([merge_data, lstm_output_df], axis=1)


def calculate_metrics(y_true, y_pred):

    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    return mse, rmse, mae, r2


def test(merge_data):
    return calculate_metrics(merge_data['Price'], merge_data['LSTM_Output'])

#mse, rmse, mae, r2
#(0.00042452628726304944, 0.02060403570330457, 0.015364266255701567, 0.9905384991197792)
merge_data_with_output = merge_data_with_output.dropna(axis=0, how='any')
print(test(merge_data_with_output))

import xgboost_run
def run_xgb(merge_data,lookback, n_estimators=20):

    train = merge_data.loc[:test_split]
    test = merge_data.loc[test_beg:]

    Lt = pd.read_csv('./ARIMA.csv')
    Lt = Lt.drop('trade_date', axis=1)

    lt_delta = test.copy()
    lt_delta.pop('0')

    checker = test.copy()
    price = checker.pop('Price')
    checker.insert(6, 'Price', price)

    lt_delta.insert(6, 'Lt', Lt)
    train_x, train_y, test_x, test_y, delta_x, delta_y, checker_x, checker_y = xgboost_run.prep_data(train, test, lt_delta, checker, lookback)
    y, yhat = xgboost_run.walk_forward_validation(train_x, test_x, test_y, n_estimators)
    return y, yhat

# y,yhat = run_xgb(merge_data_with_output, 6,10)
#
# #mse, rmse, mae, r2 (0.00011711893633394431, 0.010822150263877521, 0.00840573064930789, 0.9321160241016728)
# calculate_metrics(y, yhat)

def objective(trial):
    # Define the hyperparameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    lookback = trial.suggest_int('lookback', 1, 20)

    # Optimize the lookback and n_estimators
    y,yhat = run_xgb(merge_data_with_output, lookback, n_estimators)
    R2 = calculate_metrics(y, yhat)[-1]

    # Return the R2 score as the value to maximize
    return R2


# Create a study object and optimize the objective function
sampler = TPESampler(seed=10)
pruner = SuccessiveHalvingPruner()
study = optuna.create_study(direction='maximize',sampler=sampler, pruner=pruner)

# Provide initial parameters
initial_params = {
    'n_estimators': 10,
    'lookback': 6
}

# Enqueue the trial with initial parameters
study.enqueue_trial(initial_params)
n_trials = 50
study.optimize(lambda trial: objective(trial), n_trials=n_trials,n_jobs=-1)

# Print the best parameters
print(study.best_params)
#{'n_estimators': 84, 'lookback': 2} : 0.94

y,yhat = run_xgb(merge_data_with_output, 2,84)

#mse, rmse, mae, r2 (0.00011711893633394431, 0.010822150263877521, 0.00840573064930789, 0.9321160241016728)
#(0.00010370069940222155, 0.010183354035003474, 0.00771811462583996, 0.9455227692118909)
print(calculate_metrics(y, yhat))

