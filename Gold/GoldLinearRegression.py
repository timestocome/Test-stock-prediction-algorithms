# http://github.com/timestocome

# Attempt to predict gold prices and find outliers
# http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/TestForRandomness_RunsTest.pdf



import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 


######################################################################
# load data
########################################################################
# read in gold file
data = pd.read_csv('data/Gold_all.csv', parse_dates=True, index_col=0)
data = data[['Open']]

# convert to log values
#data['Open'] = np.log(data['Open'])


data['Open'] = pd.to_numeric(data['Open'], errors='coerce')
data['Volatility'] = data['Open'] - data['Open'].shift(1)
data = data.dropna()

gold_standard = data.loc[data.index < '01-01-1971']
gold = data.loc[data.index > '01-01-1971']



print(len(gold_standard), len(gold))
########################################################################

# try to fit linear regression model
from sklearn import linear_model

x1 = np.arange(1, len(gold)+ 1)
x2 = x1 **2
x3 = x1 **3
x4 = x1 **4     # best so far

x = [x1, x2, x3, x4]
x = np.reshape(x, (4, len(gold))).T
print(x.shape)

regression = linear_model.LinearRegression()
regression.fit(x, gold['Open'])
coeffs = regression.coef_
intercept = regression.intercept_


print(coeffs[0], coeffs[1])
gold['Regression'] = intercept + coeffs[0] * x1 + coeffs[1] * x2 + coeffs[2] * x3 + coeffs[3] * x4 
gold['Residuals'] = gold['Open'] - gold['Regression']
std_regression = gold['Regression'].std()
std_open = gold['Open'].std()


plt.figure(figsize=(18, 15))
ax1 = plt.subplot(3,1,1)
plt.plot(gold['Open'], label='Value')
plt.plot(gold['Volatility'], label='Volatility')
plt.plot(gold['Regression'] - std_regression, label='Regression - 3 std')
plt.plot(gold['Regression'], label='Regression')
plt.plot(gold['Regression'] + std_regression, label='Regression + 3 std')
plt.legend(loc='best')
plt.title('Gold Regression +/- 3 STD')



ax2 = plt.subplot(3,1,2)
plt.scatter(gold['Open'], gold['Volatility'])
plt.xlabel('Value')
plt.ylabel('Volatility')

ax3 = plt.subplot(3,1,3, sharex=ax2)
n_bins = 100
plt.hist(gold['Open'], n_bins, normed=1, histtype='bar')

plt.savefig("GoldRegression.png")
plt.show()


##################################################################
# Run's Test, part 3 of paper

gold_mean = gold['Open'].mean()
runs = gold['Open'] > gold['Regression'] 


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
print("Z (--- not random if Z > +/- 2.5)", Z1, Z2)