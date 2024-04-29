HyperOpt = False
InCol = True
Model_Type = 3
print('HyperOpt:',HyperOpt)
print('Arima + Garch:',InCol)
print('Model Type:',Model_Type)
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
torch.backends.cudnn.deterministic=True
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('brent_with_forecasted_volatility_prime.csv')
data = data.dropna(axis=0, how='any')
data.index = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data =data.drop(['Date'], axis=1)
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

cols = merge_data.columns
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
    return r2


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
    r2 = optimizer_lstm(train, test, Model_Type, hidden_dim, dropout_rate, lookback, num_epochs, batch_size, lr,
                        weight_decay)

    # Return the R2 score as the value to maximize
    return r2

if Model_Type == 1:
        best_params = {'hidden_dim': 46, 'dropout_rate': 0.22234557140321387, 'lookback': 4, 'num_epochs': 267, 'batch_size': 47, 'lr': 0.0011955493018202912, 'weight_decay': 1.3896799826651576e-05}
elif Model_Type == 2:
    best_params = {'hidden_dim': 84, 'dropout_rate': 0.10924601759225532, 'lookback': 1, 'num_epochs': 299,
                   'batch_size': 52, 'lr': 6.599320933831208e-05, 'weight_decay': 0.0001443204850539564}
else:
    best_params = {'hidden_dim': 33, 'dropout_rate': 0.45189673221289195, 'lookback': 3, 'num_epochs': 185,
                   'batch_size': 45, 'lr': 0.0020113575156252544, 'weight_decay': 1.3117765665649049e-05}

if HyperOpt:
    # Create a study object and optimize the objective function
    sampler = TPESampler(seed=10)
    pruner = SuccessiveHalvingPruner()
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=100,n_jobs=-1)

    best_params = study.best_params

model = lstm(model_type=Model_Type,input_dim=len(train.columns), hidden_dim=best_params['hidden_dim'], output_dim=1,dropout_rate=best_params['dropout_rate']).to(device)

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



def inverse_process(data,cols):
    temp = []
    for d in data:
        d = [d]
        for i in range(cols-1):
            d = d + [0]
        temp.append(d)
    return temp

#mse, rmse, mae, r2
merge_data_with_output = merge_data_with_output.dropna(axis=0, how='any')

merge_data_with_output_test = merge_data_with_output.loc[test_beg:]


model_test_inv = inverse_process(merge_data_with_output_test['Price'].values, len(merge_cols))
true_test_inv = inverse_process(merge_data_with_output_test['LSTM_Output'].values, len(merge_cols))
model_test_inv = scaler.inverse_transform(model_test_inv)[:,0]
test_test_inv = scaler.inverse_transform(true_test_inv)[:,0]



import xgboost_run
def run_xgb(merge_data,lookback, n_estimators=20):

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
    train_x, train_y, test_x, test_y, delta_x, delta_y, checker_x, checker_y = xgboost_run.prep_data(train, test, lt_delta, checker, lookback)
    y, yhat = xgboost_run.walk_forward_validation(train_x, test_x, test_y, n_estimators)
    return y, yhat


def objective(trial):
    # Define the hyperparameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 10, 100)

    # Optimize the lookback and n_estimators
    y,yhat = run_xgb(merge_data_with_output, 1, n_estimators)
    R2 = calculate_metrics(y, yhat)[-1]

    # Return the R2 score as the value to maximize
    return R2


# Create a study object and optimize the objective function
sampler = TPESampler(seed=10)
pruner = SuccessiveHalvingPruner()
study = optuna.create_study(direction='maximize',sampler=sampler, pruner=pruner)

# Provide initial parameters
initial_params = {
    'n_estimators': 10
}

# Enqueue the trial with initial parameters
study.enqueue_trial(initial_params)
n_trials = 30
study.optimize(lambda trial: objective(trial), n_trials=n_trials,n_jobs=-1)

# Print the best parameters
print(study.best_params)


y,yhat = run_xgb(merge_data_with_output, 1,study.best_params['n_estimators'])

xgb_y = np.array(inverse_process(y, len(merge_cols))).reshape(len(y),-1)
xgb_yhat = np.array(inverse_process(yhat, len(merge_cols))).reshape(len(yhat),-1)
xgb_y = scaler.inverse_transform(xgb_y)[:,0]
xgb_yhat = scaler.inverse_transform(xgb_yhat)[:,0]

print("mse, rmse, mae, r2")
print('Without XGB finetuning :',calculate_metrics(model_test_inv, test_test_inv))
print('With XGB finetuning :',calculate_metrics(xgb_y, xgb_yhat))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Create an instance of your model
model = lstm(model_type=Model_Type,
             input_dim=len(train.columns),
             hidden_dim=best_params['hidden_dim'],
             output_dim=1,
             dropout_rate=best_params['dropout_rate']).to(device)

# Compute the total number of parameters in the model
total_params = count_parameters(model)
print(f'Total number of parameters in the model: {total_params}')


