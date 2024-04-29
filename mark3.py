InCol = True
HyperOpt = False
print('Arima + Garch:',InCol)
print('HyperOpt:',HyperOpt)
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:32'
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba import Mamba, MambaConfig
torch.backends.cudnn.deterministic=True
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

brent = pd.read_csv('brent_with_forecasted_volatility_prime.csv')
data = pd.read_excel('brent_vec.xlsx')
data = data.dropna(axis=0, how='any')
data.index = pd.to_datetime(brent['Date'], format='%m/%d/%Y')
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

merge_data['pre_close'] = merge_data['Price'].shift(1)
merge_data = merge_data.dropna(axis=0, how='any')

merge_data['Price_pre_close_pct_change'] = ((merge_data['Price'] - merge_data['pre_close']) / merge_data['pre_close']) * 100
merge_idx = merge_data.index
merge_cols = merge_data.columns


merge_data['Price_pre_close_pct_change'] = merge_data['Price_pre_close_pct_change'].apply(lambda x:0.01*x).values
merge_data.pop('pre_close')


train = merge_data.loc[:test_split]
test = merge_data.loc[test_beg:]
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
train_price = train.pop('Price')
test_price = test.pop('Price')


class Net(nn.Module):
    def __init__(self, in_dim, out_dim,hidden,layer):
        super().__init__()
        self.config = MambaConfig(d_model=hidden, n_layers=layer)
        self.mamba = nn.Sequential(
            nn.Linear(in_dim, hidden),
            Mamba(self.config),
            nn.Linear(hidden, out_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.mamba(x)
        return x.flatten()


def evaluation_metric(y_test,y_hat):
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = mean_absolute_error(y_test,y_hat)
    R2 = r2_score(y_test,y_hat)
    return MSE,RMSE,MAE,R2

def evluate_model(predictions,test_price):
    temp = len(test_price)
    final_pred = []
    for i in range(temp):
        pred = merge_data['Price'].values[-temp-1+i]*(1+predictions[i])
        final_pred.append(pred)
    return evaluation_metric(test_price, final_pred)

def PredictLoss(trainX, trainy, testX,test_price,hidden=16,epoch=100,lr=0.01,weight_decay=0.01,num_layers=2):
    clf = Net(len(trainX[0]), 1,hidden,num_layers)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    xt = torch.from_numpy(trainX).float().unsqueeze(0)
    xv = torch.from_numpy(testX).float().unsqueeze(0)
    yt = torch.from_numpy(trainy).float()
    if True:
        clf = clf.cuda()
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()
    for e in range(epoch):
        clf.train()
        z = clf(xt)
        loss = F.mse_loss(z, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    clf.eval()
    mat = clf(xv)
    if True: mat = mat.cpu()
    yhat = mat.detach().numpy().flatten()
    return evluate_model(yhat,test_price)

def PredictWithData(trainX, trainy, testX,hidden=16,epoch=100,lr=0.01,weight_decay=0.01,num_layers=2):
    clf = Net(len(trainX[0]), 1,hidden,num_layers)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    xt = torch.from_numpy(trainX).float().unsqueeze(0)
    xv = torch.from_numpy(testX).float().unsqueeze(0)
    yt = torch.from_numpy(trainy).float()
    if True:
        clf = clf.cuda()
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()
    for e in range(epoch):
        clf.train()
        z = clf(xt)
        loss = F.mse_loss(z, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e % 10 == 0 and e != 0:
            print('Epoch %d | Lossp: %.4f' % (e, loss.item()))


    clf.eval()
    mat = clf(xv)
    if True: mat = mat.cpu()
    yhat = mat.detach().numpy().flatten()
    return yhat

trainy = train.pop('Price_pre_close_pct_change').values
testy = test.pop('Price_pre_close_pct_change').values
trainX = train.values
testX = test.values

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

net = Net(len(trainX[0]), 1, 16, 2)

total_params = count_parameters(net)
print(f'Total number of parameters in the model: {total_params}')


def objective(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 1, 25)
    num_epochs = trial.suggest_int('num_epochs', 50, 300)
    num_layers = trial.suggest_int('num_epochs', 1, 5)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)

    r2 = PredictLoss(trainX, trainy, testX,test_price,hidden=hidden_dim,epoch=num_epochs,lr=lr,weight_decay=weight_decay,num_layers=num_layers)

    return r2


best_param = {'hidden_dim': 16, 'num_epochs': 100, 'num_layer': 2, 'lr': 0.01, 'weight_decay': 0.0001}

if HyperOpt:
    # Define the initial parameters
    initial_params = {'hidden_dim': 10, 'num_epochs': 255, 'num_layer': 1, 'lr': 0.00372006824710135, 'weight_decay': 0.00025504132602412996}

    # Create a study object and optimize the objective function
    sampler = TPESampler(seed=10)
    pruner = SuccessiveHalvingPruner()
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.enqueue_trial(initial_params)
    study.optimize(objective, n_trials=100)


    best_param = study.best_params



R2_l = []
MSE_l = []
MAE_l = []
RMSE_l = []
for i in tqdm(range(10)):
    MSE,RMSE,MAE,R2 = PredictLoss(trainX, trainy, testX,test_price,hidden=best_param['hidden_dim'],epoch=best_param['num_epochs'],lr=best_param['lr'],weight_decay=best_param['weight_decay'],num_layers=best_param['num_layer'])
    R2_l.append(R2)
    MSE_l.append(MSE)
    MAE_l.append(MAE)
    RMSE_l.append(RMSE)

print("mse, rmse, mae, r2")
print(sum(MSE_l)/10,end=' ')
print(sum(RMSE_l)/10,end=' ')
print(sum(MAE_l)/10,end=' ')
print(sum(R2_l)/10,end=' ')

