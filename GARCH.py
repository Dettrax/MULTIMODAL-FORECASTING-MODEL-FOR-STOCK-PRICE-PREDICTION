import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from utils import *
from arch import arch_model

data = pd.read_csv('brent_processed.csv')
data = data.dropna(axis=0, how='any')
print(data.isnull().any())

data.index = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data = data.drop(['Date'], axis=1)
data = pd.DataFrame(data, dtype=np.float64)

data.sort_index(inplace=True)

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

model = sm.tsa.ARIMA(endog=data['Price'], order=(2, 1, 0)).fit()

garch_result = arch_model(model.resid, vol='Garch', p=1,q=1).fit(disp='off')

forecasts = garch_result.forecast(start=0)

print("Forecasted Mean:")
print(forecasts.mean)
print("\nForecasted Variance:")
print(forecasts.variance)

forecasted_volatility = np.sqrt(forecasts.variance)

forecasted_volatility.index = data.index

data = data.assign(Forecasted_Volatility=forecasted_volatility)

data.to_csv('brent_with_forecasted_volatility.csv')