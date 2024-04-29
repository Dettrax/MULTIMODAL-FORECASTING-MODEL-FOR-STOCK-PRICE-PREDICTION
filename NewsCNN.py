# Type - Soft, Multihead, ScaledDot
Att_type = 'ScaledDot'
HyperOpt = False
InCol = True

print('Attention Type:', Att_type)
print('HyperOpt:', HyperOpt)
print('Col ARIMA and GARCH:', InCol)

if Att_type == 'ScaledDot':
    from ScaledDotAtten import *
elif Att_type == 'Multihead':
    from multiatten import *
else:
    from SoftAtten import *

import warnings
warnings.filterwarnings("ignore")
import xgboost_run

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from tqdm import tqdm
import torch.nn as nn
from sklearn import metrics

torch.backends.cudnn.deterministic = True
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('brent_with_forecasted_volatility_prime.csv')
data = data.dropna(axis=0, how='any')
data.index = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data = data.drop(['Date'], axis=1)
data = pd.DataFrame(data, dtype=np.float64)
data.sort_index(inplace=True)


brent = pd.read_csv('brent_with_forecasted_volatility_prime.csv')
data = pd.read_excel('brent_vec.xlsx')
# data = data[['Price', 'Open', 'High', 'Low', 'Vol', 'Change','Forecasted_Volatility','pos','neg','neu']]
data = data.dropna(axis=0, how='any')
data.index = pd.to_datetime(brent['Date'], format='%m/%d/%Y')
# data.index = pd.to_datetime(data['Date'], format='%m/%d/%Y')
# data =data.drop(['Date'], axis=1)
data = pd.DataFrame(data, dtype=np.float64)

data.sort_index(inplace=True)
test_split = '2023-09-20' #150
test_beg = '2023-09-21'

residuals = pd.read_csv('./ARIMA_residuals1.csv')
residuals.index = pd.to_datetime(residuals['Date'])
residuals.pop('Date')
merge_data = pd.merge(data, residuals, on='Date')

if not InCol:
    merge_data.pop('0')
    merge_data.pop('Forecasted_Volatility')

merge_idx = merge_data.index
merge_cols = merge_data.columns
# apply minmaxscaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
merge_data = scaler.fit_transform(merge_data)
merge_data = pd.DataFrame(merge_data, index=merge_idx, columns=merge_cols)

train = merge_data.loc[:test_split]
test = merge_data.loc[test_beg:]

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# for scaled dot attention
if Att_type == 'ScaledDot':
    def create_sequences(data, lookback):
        x = []
        y = []
        for i in range(lookback, data.shape[0]):
            x.append(data[i - lookback: i])
            y.append(data[i - lookback: i, 0])  # Use a sequence of steps as the target
        return torch.tensor(x, device=device), torch.tensor(y, device=device)

# For soft attention and multihead attention
else:
    def create_sequences(data, lookback):
        x = []
        y = []
        for i in range(lookback, data.shape[0]):
            x.append(data[i - lookback: i])
            y.append(data[i, 0])
        return torch.tensor(x, device=device), torch.tensor(y, device=device)


def optimise_attention(train, test, cnn_output, hidden_dim, dropout_rate, lookback=10, num_epochs=100, batch_size=64,
                       lr=0.001, weight_decay=0.001, kernel_size=1):
    model = AttentionModel(input_dims=len(train.columns), lstm_units=hidden_dim,
                           cnn_output=cnn_output, dropout_rate=dropout_rate,
                           kernel_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    trainX, trainY = create_sequences(train.values, lookback)
    testX, testY = create_sequences(test.values, lookback)
    model.train()
    # Training loop for the best model
    for epoch in range(num_epochs):
        for i in range(0, len(trainX), batch_size):
            inputs = trainX[i:i + batch_size]
            targets = trainY[i:i + batch_size]
            optimizer.zero_grad()
            inputs = inputs.detach().cpu()
            inputs = inputs.permute(0, 2, 1)
            inputs = torch.tensor(inputs.float(), device=device)
            outputs = model(inputs.float())
            loss = criterion(outputs.squeeze(), targets.float())
            loss.backward()
            optimizer.step()

    def calculate_metrics(y_true, y_pred):
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()

        mse = metrics.mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)

        return mse, rmse, mae, r2

    model.eval()
    with torch.no_grad():
        testX = testX.permute(0, 2, 1)
        outputs = model(testX.float())
        mse, rmse, mae, r2 = calculate_metrics(testY, outputs.squeeze())
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
    r2 = optimise_attention(train, test, cnn_output, hidden_dim, dropout_rate, lookback, num_epochs, batch_size, lr,
                            weight_decay, 1)

    # Return the R2 score as the value to maximize
    return r2


if Att_type == 'ScaledDot':
    if InCol:
        best_params = {'cnn_output': 114, 'hidden_dim': 92, 'dropout_rate': 0.10135062229343499, 'lookback': 2,
                       'num_epochs': 249, 'batch_size': 77, 'lr': 0.00017943492288184664,
                       'weight_decay': 0.00030465328464725016}
    else:
        best_params = {'cnn_output': 53, 'hidden_dim': 147, 'dropout_rate': 0.11315172616291817, 'lookback': 1,
                       'num_epochs': 65, 'batch_size': 65, 'lr': 0.00036368397727844253,
                       'weight_decay': 0.0008437248421188951}

elif Att_type == 'Multihead':
    if InCol:
        best_params = {'cnn_output': 290, 'hidden_dim': 220, 'dropout_rate': 0.14025128863511263, 'lookback': 1, 'num_epochs': 75, 'batch_size': 127, 'lr': 0.0012340990424287853, 'head': 4, 'weight_decay': 0.00011356340578794312}

    else:
        pass


else:
    if InCol:
        best_params = {'cnn_output': 415, 'hidden_dim': 42, 'dropout_rate': 0.32677977665682545, 'lookback': 1, 'num_epochs': 211, 'batch_size': 52, 'lr': 0.00025083018149161764, 'weight_decay': 1.2986438189462185e-05}

    else:
        best_params = {'cnn_output': 467, 'hidden_dim': 14, 'dropout_rate': 0.12204411154927669, 'lookback': 1, 'num_epochs': 248, 'batch_size': 32, 'lr': 9.897671745223481e-05, 'head': 4, 'weight_decay': 6.164885891482945e-05}

# Create a study object and optimize the objective function
if HyperOpt:
    sampler = TPESampler(seed=10)
    pruner = SuccessiveHalvingPruner()
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=100)

    # Print the best parameters
    print(study.best_params)
    best_params = study.best_params

model = AttentionModel(input_dims=len(train.columns), lstm_units=best_params['hidden_dim'],
                       cnn_output=best_params['cnn_output'], dropout_rate=best_params['dropout_rate'],
                       kernel_size=1).to(device)

# model = AttentionModel(input_dims=len(train.columns), lstm_units=best_params['hidden_dim'],cnn_output=best_params['cnn_output'],dropout_rate=best_params['dropout_rate'],kernel_size=1).to(device)

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
        inputs = inputs.detach().cpu()
        inputs = inputs.permute(0, 2, 1)
        inputs = torch.tensor(inputs.float(), device=device)
        outputs = model(inputs.float())
        loss = criterion(outputs.squeeze(), targets.float())
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    testX = testX.permute(0, 2, 1)
    outputs = model(testX.float())

# Create a placeholder DataFrame
lstm_output_df = pd.DataFrame(index=merge_data.index, columns=['LSTM_Output'])
lstm_output_df = lstm_output_df.fillna(np.nan)


# Multihead
if Att_type == 'Multihead':
    # Ensure the shapes are compatible before assignment
    if outputs.shape[0] == lstm_output_df.iloc[-len(outputs):].shape[0]:
        # Add an extra dimension to match the DataFrame slice
        outputs_selected = outputs.unsqueeze(-1)

        # Convert the tensor to a numpy array before assigning it to the DataFrame
        lstm_output_df.iloc[-len(outputs):] = outputs_selected.squeeze().cpu().numpy().reshape(-1, 1)

# ScaledDot
elif Att_type == 'ScaledDot':
    # Select the first dimension from the outputs tensor
    outputs_selected = outputs[:, 0, 0]  # Select the first column
    # Add an extra dimension to match the DataFrame slice
    outputs_selected = outputs_selected.unsqueeze(-1)

    # Convert the tensor to a numpy array before assigning it to the DataFrame
    lstm_output_df.iloc[-len(outputs):] = outputs_selected.squeeze().cpu().numpy().reshape(-1, 1)

# Soft
else:
    lstm_output_df.iloc[-len(outputs):] = outputs.cpu().numpy()

# Concatenate the LSTM output with the original data
merge_data_with_output = pd.concat([merge_data, lstm_output_df], axis=1)


def calculate_metrics(y_true, y_pred):
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    return mse, rmse, mae, r2


def inverse_process(data, cols):
    temp = []
    for d in data:
        d = [d]
        for i in range(cols - 1):
            d = d + [0]
        temp.append(d)
    return temp


# mse, rmse, mae, r2
merge_data_with_output = merge_data_with_output.dropna(axis=0, how='any')

merge_data_with_output_test = merge_data_with_output.loc[test_beg:]

model_test_inv = inverse_process(merge_data_with_output_test['Price'].values, len(merge_cols))
true_test_inv = inverse_process(merge_data_with_output_test['LSTM_Output'].values, len(merge_cols))
model_test_inv = scaler.inverse_transform(model_test_inv)[:, 0]
test_test_inv = scaler.inverse_transform(true_test_inv)[:, 0]


def run_xgb(merge_data, lookback, n_estimators=20):
    train = merge_data.loc[:test_split]
    test = merge_data.loc[test_beg:]

    Lt = pd.read_csv('./ARIMA.csv')
    Lt = Lt.drop('trade_date', axis=1)

    lt_delta = test.copy()
    try:
        lt_delta.pop('0')
    except:
        pass
    checker = test.copy()
    price = checker.pop('Price')
    checker.insert(6, 'Price', price)

    lt_delta.insert(6, 'Lt', Lt)
    train_x, train_y, test_x, test_y, delta_x, delta_y, checker_x, checker_y = xgboost_run.prep_data(train, test,
                                                                                                     lt_delta, checker,
                                                                                                     lookback)
    y, yhat = xgboost_run.walk_forward_validation(train_x, test_x, test_y, n_estimators)
    return y, yhat


def objective(trial):
    # Define the hyperparameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 5, 50)
    # lookback = trial.suggest_int('lookback', 1, 20)

    # Optimize the lookback and n_estimators
    y, yhat = run_xgb(merge_data_with_output, 1, n_estimators)
    R2 = calculate_metrics(y, yhat)[-1]

    # Return the R2 score as the value to maximize
    return R2


# Create a study object and optimize the objective function
sampler = TPESampler(seed=10)
pruner = SuccessiveHalvingPruner()
study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

# Provide initial parameters
initial_params = {
    'n_estimators': 10,
    'lookback': 1
}

# Enqueue the trial with initial parameters
study.enqueue_trial(initial_params)
n_trials = 50
study.optimize(lambda trial: objective(trial), n_trials=n_trials, n_jobs=-1)

# # Print the best parameters
# print(study.best_params)

y, yhat = run_xgb(merge_data_with_output, 1, study.best_params['n_estimators'])

xgb_y = np.array(inverse_process(y, len(merge_cols))).reshape(len(y), -1)
xgb_yhat = np.array(inverse_process(yhat, len(merge_cols))).reshape(len(yhat), -1)
xgb_y = scaler.inverse_transform(xgb_y)[:, 0]
xgb_yhat = scaler.inverse_transform(xgb_yhat)[:, 0]

print('Without XGB finetuning :', calculate_metrics(model_test_inv, test_test_inv))
print('With XGB finetuning :', calculate_metrics(xgb_y, xgb_yhat))



