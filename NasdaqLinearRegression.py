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


plt.figure(figsize=(18, 15))
ax1 = plt.subplot(3,1,1)
plt.plot(data['Open'], label='Value')
plt.plot(data['Volatility'], label='Volatility')
plt.plot(data['Regression'] - std_regression, label='Regression - std')
plt.plot(data['Regression'], label='Regression')
plt.plot(data['Regression'] + std_regression, label='Regression + std')
plt.legend(loc='best')
plt.title('Nasdaq Regression +/-  STD')



ax2 = plt.subplot(3,1,2)
plt.scatter(data['Open'], data['Volatility'])
plt.xlabel('Value')
plt.ylabel('Volatility')

ax3 = plt.subplot(3,1,3, sharex=ax2)
n_bins = 100
plt.hist(data['Open'], n_bins, normed=1, histtype='bar')

plt.savefig("dataRegression.png")
plt.show()


##################################################################
# Run's Test, part 3 of paper

data_mean = data['Open'].mean()
runs = data['Open'] > data['Regression'] 


# convert runs data to number of runs 
R = 0
r_prev = runs[0]
for r in runs:
    if r != r_prev: R += 1
    r_prev = r


T = len(runs)
Ta = runs.sum()
Tb = T - Ta 

E = (T + 2 * Ta * Tb) / T           # expected runs
V = (2 * Ta * Tb * (2*Ta*Tb - T)) / (T **2 * (T - 1))  # variance of runs

Z1 = (R - E) / std_open
Z2 = (R -E) / std_regression
print("Run's Test Results")
print("R %lf, E %lf, V %lf" % (R, E, V))
print("Z (not random if Z > +/- 2.5)", Z1, Z2)