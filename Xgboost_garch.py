import pandas as pd
import numpy as np
import xgboost_run as xgb
from sklearn import metrics
# import warnings
# warnings.filterwarnings("ignore")

from tqdm import tqdm
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
data = pd.read_csv('brent_vol.csv')


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



def create_sequences(data, lookback):
    x = []
    y = []
    for i in range(lookback, data.shape[0]):
        x.append(data[i - lookback: i].flatten())
        y.append(data[i, -1])
    return np.array(x), np.array(y)

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

def prep_data(train,test,lt,checker,lookback):
    train_x , train_y = create_sequences(train.values, lookback)
    test_x, test_y = create_sequences(test.values, lookback)
    delta_x, delta_y = create_sequences(lt.values, lookback)
    checker_x , checker_y = create_sequences(checker.values, lookback)
    return train_x, train_y, test_x, test_y, delta_x, delta_y, checker_x, checker_y

# train_x, train_y, test_x, test_y, delta_x, delta_y, checker_x, checker_y = prep_data(train,test,lt,checker,2)

def xgboost_forecast(train, testX, n_estimators=20):
    # transform list into array
    train = np.asarray(train)
    # print('train', train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # print('trainX', trainX, 'trainy', trainy)
    # fit model

    # Assuming 'data' is your input data and 'label' is your target variable
    gpy_x = xgb.DMatrix(trainX, label=trainy)
    # Then use 'dtrain' in your XGBoost model
    param = {'objective': 'reg:squarederror',
             'tree_method': 'hist',
             'device': 'gpu'
             }
    num_boost_round = n_estimators
    model = xgb.train(param, gpy_x, num_boost_round=num_boost_round)
    # model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, tree_method='gpu_hist',device='cuda')
    # model.fit(trainX)
    # make a one-step prediction
    testX_dmatrix = xgb.DMatrix(np.asarray([testX]))
    yhat = model.predict(testX_dmatrix)
    return yhat[0]

def walk_forward_validation(train_x,test_x,test_y,n_estimators=20):
    predictions = list()
    train = train_x
    history = [x for x in train]
    # print('history', history)
    test_x = np.asarray(test_x)
    for i in range(len(test_x)):
        testX, testy = test_x[i , :-1], test_y[i]
        # print('i', i, testX, testy)
        yhat = xgboost_forecast(history, testX, n_estimators)
        predictions.append(yhat)
        history.append(test_x[i, :])
        # print(i+1, '>expected=%.6f, predicted=%.6f' % (testy, yhat))
    return test_y,predictions


def evaluation_metric(y_test,y_hat):
    MSE = metrics.mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = metrics.mean_absolute_error(y_test,y_hat)
    R2 = metrics.r2_score(y_test,y_hat)
    print('MSE: %.5f' % MSE)
    print('RMSE: %.5f' % RMSE)
    print('MAE: %.5f' % MAE)
    print('R2: %.5f' % R2)
    return R2

def prediction(train_x,test_x,test_y,checker_y,delta_y,n_estimators=20):
    y ,yhat = walk_forward_validation(train_x,test_x,test_y,n_estimators)
    finalpredicted_stock_price = [i + j for i, j in zip(delta_y, yhat)]
    return evaluation_metric(checker_y, finalpredicted_stock_price)



def optimise_loookback(train,test,lt,checker,lookback, n_estimators=20):
    train_x, train_y, test_x, test_y, delta_x, delta_y, checker_x, checker_y = prep_data(train, test, lt, checker, lookback)
    R2 = prediction(train_x, test_x, test_y, checker_y, delta_y, n_estimators)
    return R2

def objective(trial):
    # Define the hyperparameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    lookback = trial.suggest_int('lookback', 1, 20)

    # Optimize the lookback and n_estimators
    R2 = optimise_loookback(train, test, lt_delta, checker, lookback, n_estimators)
    # Return the R2 score as the value to maximize
    return R2

def progress_bar(n_trials, study, trial):
    progress = tqdm(total=n_trials, ncols=80, position=0, leave=True)
    progress.update(trial.number + 1 - progress.n)
    if trial.number + 1 == n_trials:
        progress.close()

# Create a study object and optimize the objective function
sampler = TPESampler(seed=10)
pruner = SuccessiveHalvingPruner()
study = optuna.create_study(direction='maximize',sampler=sampler, pruner=pruner)

# Provide initial parameters
initial_params = {
    'n_estimators': 20,
    'lookback': 6
}

# Enqueue the trial with initial parameters
study.enqueue_trial(initial_params)
n_trials = 50
study.optimize(lambda trial: objective(trial), n_trials=n_trials,n_jobs=-1, callbacks=[lambda study, trial: progress_bar(n_trials, study, trial)])

# Print the best parameters
print(study.best_params)

# R2 = optimise_loookback(train, test, lt_delta, checker, 16, 17)

# MSE: 2.09003
# RMSE: 1.44569
# MAE: 1.11271
# R2: 0.90015

R2 = optimise_loookback(train, test, lt_delta, checker, 11, 12)

# Garch
# MSE: 0.00019
# RMSE: 0.01390
# MAE: 0.01070
# R2: 0.88985


#Xgboost
# MSE: 0.00017
# RMSE: 0.01320
# MAE: 0.01045
# R2: 0.90004

#Arima
# MSE: 0.00021
# RMSE: 0.01444
# MAE: 0.01073
# R2: 0.90514

