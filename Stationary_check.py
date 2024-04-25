import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from utils import *
# Use the 'openpyxl' engine to read the Excel file
data = pd.read_csv('brent.csv')
#drop nan value in all the row
data = data.dropna(axis=0, how='any')

#check if any column has nan value
print(data.isnull().any())

#My volume column is in string format like 477.54k , 699.9k, so I will convert it to float
def convert_k_to_float(value):
    if 'k' in value or 'K' in value:
        return float(value.replace('k', '').replace('K', '')) * 1000
    return float(value)

# Apply the function to the 'volume' column
data['Vol'] = data['Vol'].apply(convert_k_to_float)

#change column is also string like 0.21% , 0.25%, so I will convert it to float

def convert_percent_to_float(value):
    if '%' in value:
        return float(value.replace('%', ''))
    return float(value)

# Apply the function to the 'change' column
data['Change'] = data['Change'].apply(convert_percent_to_float)

def perform_adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])

perform_adf_test(data['Price'])
#0.65 > 0.005, so it is not stationary

#take the first order difference
data['Price_diff'] = data['Price'].diff(1)
data = data.dropna(axis=0, how='any')
perform_adf_test(data['Price_diff'])
#0 < 0.005, so it is stationary

perform_adf_test(data['Change'])
#0 < 0.005, so it is stationary

data.index = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data =data.drop(['Date','Change','Price_diff'], axis=1)
data = pd.DataFrame(data, dtype=np.float64)

#Creating test set with 0.8 ratio
# Calculate the split index
split_index = int(len(data) * 0.1)

# Split the data into training and test sets
train = data.iloc[split_index:]
test = data.iloc[:split_index]

plt.figure(figsize=(10, 6))
plt.plot(train['Price'], label='training_set')
plt.plot(test['Price'], label='test_set')
plt.title('Close price')
plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()

temp = np.array(train['Price'])

# First-order diff
train['diff_1'] = train['Price'].diff(1)
plt.figure(figsize=(10, 6))
train['diff_1'].plot()
plt.title('First-order diff')
plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('diff_1', fontsize=14, horizontalalignment='center')
plt.show()



# Second-order diff
train['diff_2'] = train['diff_1'].diff(1)
plt.figure(figsize=(10, 6))
train['diff_2'].plot()
plt.title('Second-order diff')
plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('diff_2', fontsize=14, horizontalalignment='center')
plt.show()

temp1 = np.diff(train['Price'], n=1)

# white noise test
training_data1 = train['Price'].diff(1)
# training_data1_nona = training_data1.dropna()
temp2 = np.diff(train['Price'], n=1)
# print(acorr_ljungbox(training_data1_nona, lags=2, boxpierce=True, return_df=True))
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
# print('history', type(history), history)
predictions = list()
# print('test_set.shape', test_set.shape[0])
for t in range(test.shape[0]):
    model1 = sm.tsa.ARIMA(history, order=(2, 1, 0))
    model_fit = model1.fit()
    yhat = model_fit.forecast()
    yhat = np.float(yhat[0])
    predictions.append(yhat)
    obs = test.iloc[t, 0]
    # obs = np.float(obs)
    # print('obs', type(obs))
    history.append(obs)
    # print(test_set.index[t])
    # print(t+1, 'predicted=%f, expected=%f' % (yhat, obs))
#print('predictions', predictions)

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


predictions_ARIMA_diff = pd.Series(model.fittedvalues, copy=True)
predictions_ARIMA_diff = predictions_ARIMA_diff[2600:]
print('#', predictions_ARIMA_diff)
plt.figure(figsize=(10, 6))
plt.plot(training_data_diff, label="diff_1")
plt.plot(predictions_ARIMA_diff, label="prediction_diff_1")
plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('diff_1', fontsize=14, horizontalalignment='center')
plt.title('DiffFit')
plt.legend()
plt.show()