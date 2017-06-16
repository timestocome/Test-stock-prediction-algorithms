# http://github.com/timestocome

# Attempt to predict nasdaq indexes and find outliers
# http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/TestForRandomness_RunsTest.pdf



import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 


######################################################################
# load data
########################################################################
# read in data file
data = pd.read_csv('data/nasdaq.csv', parse_dates=True, index_col=0)
data = data[['Open']]

data['Open'] = pd.to_numeric(data['Open'], errors='coerce')

# convert to log values
#data['Open'] = np.log(data['Open'])


data['Open'] = pd.to_numeric(data['Open'], errors='coerce')
data['Volatility'] = data['Open'] - data['Open'].shift(1)
data = data.dropna()


########################################################################

# try to fit linear regression model
from sklearn import linear_model

x1 = np.arange(1, len(data)+ 1)
x2 = x1 **2
x3 = x1 **3
x4 = x1 **4     # best so far

x = [x1, x2, x3, x4]
x = np.reshape(x, (4, len(data))).T
print(x.shape)

regression = linear_model.LinearRegression()
regression.fit(x, data['Open'])
coeffs = regression.coef_
intercept = regression.intercept_


print(coeffs[0], coeffs[1])
data['Regression'] = intercept + coeffs[0] * x1 + coeffs[1] * x2 + coeffs[2] * x3 + coeffs[3] * x4 
data['Residuals'] = data['Open'] - data['Regression']
std_regression = data['Regression'].std()
std_open = data['Open'].std()


p1 = np.arange(1, len(data)+ 253)
p2 = p1 **2
p3 = p1 **3
p4 = p1 **4    
past_future = intercept + coeffs[0] * p1 + coeffs[1] * p2 + coeffs[2] * p3 + coeffs[3] * p4 

'''
plt.figure(figsize=(18, 15))
plt.plot(data['Open'], label='Value')
plt.plot(data['Volatility'], label='Volatility')
plt.plot(data['Regression'] - std_regression, label='Regression - std')
plt.plot(data['Regression'], label='Regression')
plt.plot(data['Regression'] + std_regression, label='Regression + std')
plt.legend(loc='best')
plt.title('Nasdaq Regression +/-  STD')
plt.show()
'''

#######################################################################
# predict next 12 months ~253 trading days

dates = pd.bdate_range('1971-01-01', '2018-12-31')


x1 = np.arange(1, len(dates) + 1)
x2 = x1 **2
x3 = x1 **3
x4 = x1 **4  

nasdaq_futures = intercept + coeffs[0] * x1 + coeffs[1] * x2 + coeffs[2] * x3 + coeffs[3] * x4 
std_regression = data['Regression'].std()


predicted = pd.DataFrame(data=nasdaq_futures, index=dates)
predicted.index.name = 'Date'
predicted.columns = ['Open']


actual = pd.read_csv('data/nasdaq.csv', parse_dates=True, index_col=0)
actual['Open'] = pd.to_numeric(actual['Open'], errors='coerce')
actual = actual['Open']

plt.figure(figsize=(18, 16))
plt.plot(actual, label="Actual")
plt.plot(predicted, label="Predicted")
plt.plot(predicted - std_regression, label='Predicted - std')
plt.plot(predicted + std_regression, label='Predicted + std')
plt.legend(loc='best')
plt.title("Nasdaq 1971 - predicted 2019")
plt.savefig("Nasdaq_Predictions_2018.png")
plt.show()
