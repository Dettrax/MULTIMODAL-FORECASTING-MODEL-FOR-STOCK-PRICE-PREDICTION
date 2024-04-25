from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

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


split_index = int(len(data) * 0.1)

# Split the data into training and test sets
train = data.iloc[split_index:]
test = data.iloc[:split_index]

# Fit ARIMA model
model = sm.tsa.ARIMA(endog=data['Price'], order=(2, 1, 0)).fit()

# Perform LM test
lm_test = het_arch(model.resid)

# Print the results of the LM test
print('Lagrange Multiplier statistic: ', lm_test[0])
print('p-value: ', lm_test[1])

# Fit a GARCH(1, 1) model to the residuals of the ARIMA model
garch_model = arch_model(model.resid, vol='Garch', p=1, q=1)

# Fit the GARCH model
garch_result = garch_model.fit()

# Print the summary of the GARCH model
print(garch_result.summary())

# Forecast the volatility for all observations
forecasts = garch_result.forecast(start=0)

# # Print the forecasted mean and variance
# print("Forecasted Mean:")
# print(forecasts.mean)
# print("\nForecasted Variance:")
# print(forecasts.variance)

forecasted_volatility = np.sqrt(forecasts.variance)

# Ensure the forecasted volatility has the same index as your data DataFrame
forecasted_volatility.index = data.index

# Add the forecasted volatility as a new column to the DataFrame
data = data.assign(Forecasted_Volatility=forecasted_volatility)

# Now, 'data' DataFrame has an additional column 'Forecasted_Volatility'

data.to_csv('brent_with_forecasted_volatility.csv', index=False)