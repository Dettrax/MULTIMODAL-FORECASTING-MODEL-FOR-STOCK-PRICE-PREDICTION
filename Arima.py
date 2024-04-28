import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from utils import *

data = pd.read_csv('brent.csv')
data = data.dropna(axis=0, how='any')
print(data.isnull().any())

def convert_k_to_float(value):
    if 'k' in value or 'K' in value:
        return float(value.replace('k', '').replace('K', '')) * 1000
    return float(value)

data['Vol'] = data['Vol'].apply(convert_k_to_float)

def convert_percent_to_float(value):
    if '%' in value:
        return float(value.replace('%', ''))
    return float(value)

data['Change'] = data['Change'].apply(convert_percent_to_float)

data.index = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data =data.drop(['Date','Change'], axis=1)
data = pd.DataFrame(data, dtype=np.float64)

data.sort_index(inplace=True)
data_idx = data.index
#apply minmaxscaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data, index=data_idx, columns=['Price', 'Open', 'High', 'Low', 'Vol'])


test_split = '2023-09-20'
test_beg = '2023-09-21'

train = data.loc[:test_split]
test = data.loc[test_beg:]

plt.figure(figsize=(10, 6))
plt.plot(train['Price'], label='training_set')
plt.plot(test['Price'], label='test_set')
plt.title('Close price')
plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()

temp = np.array(train['Price'])

train['diff_1'] = train['Price'].diff(1)
plt.figure(figsize=(10, 6))
train['diff_1'].plot()
plt.title('First-order diff')
plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('diff_1', fontsize=14, horizontalalignment='center')
plt.show()

train['diff_2'] = train['diff_1'].diff(1)
plt.figure(figsize=(10, 6))
train['diff_2'].plot()
plt.title('Second-order diff')
plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('diff_2', fontsize=14, horizontalalignment='center')
plt.show()

temp1 = np.diff(train['Price'], n=1)

training_data1 = train['Price'].diff(1)
temp2 = np.diff(train['Price'], n=1)
print(acorr_ljungbox(temp2, lags=2, boxpierce=True))

acf_pacf_plot(train['Price'],acf_lags=500)

price = list(temp2)
data2 = {
    'trade_date': train['diff_1'].index[1:],
    'close': price
}

df = pd.DataFrame(data2)
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%m/%d/%Y')

training_data_diff = df.set_index(['trade_date'], drop=True)

acf_pacf_plot(training_data_diff)

model = sm.tsa.ARIMA(endog=train['Price'], order=(2, 1, 0)).fit()
print(model.summary())

history = [x for x in train['Price']]
predictions = list()
for t in range(test.shape[0]):
    model1 = sm.tsa.ARIMA(history, order=(2, 1, 0))
    model_fit = model1.fit()
    yhat = model_fit.forecast()
    yhat = np.float(yhat[0])
    predictions.append(yhat)
    obs = test.iloc[t, 0]
    history.append(obs)

predictions1 = {
    'trade_date': test.index[:],
    'close': predictions
}
predictions1 = pd.DataFrame(predictions1)
predictions1 = predictions1.set_index(['trade_date'], drop=True)
predictions1.to_csv('./ARIMA.csv')
plt.figure(figsize=(10, 6))
plt.plot(test['Price'], label='Stock Price')
plt.plot(predictions1, label='Predicted Stock Price')
plt.title('ARIMA: Stock Price Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()

model2 = sm.tsa.ARIMA(endog=data['Price'], order=(2, 1, 0)).fit()
residuals = pd.DataFrame(model2.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
residuals.to_csv('./ARIMA_residuals1.csv')
evaluation_metric(test['Price'],predictions)
adf_test(temp)
adf_test(temp1)

data.to_csv('brent_processed.csv')