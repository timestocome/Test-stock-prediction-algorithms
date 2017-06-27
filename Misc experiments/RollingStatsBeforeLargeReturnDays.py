# http://github.com/timestocome

# look at best worst returns days
# see which, if any, stats change in a rolling window leading up 
# to best/worst days


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# pandas display options
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 25
pd.options.display.width = 1000

######################################################################
# data
########################################################################

# read in datafile
data = pd.read_csv('data/nasdaq.csv', index_col=0, parse_dates=True)
data = data.dropna(axis=0)      # ditch nulls
data = data[['Open']]           # ditch the stuff not being used
data['Open'] = pd.to_numeric(data['Open'], errors='coerce')  # convert string to numbers

# switch to log data
data['logNASDAQ'] = np.log(data['Open'])

# log returns
data['logReturns'] = data['logNASDAQ'] - data['logNASDAQ'].shift(1)

# drop NaN
data.dropna()


# remove nan row from target creation
data = data.dropna()

def rolling_stats(w):

    window_shift = int(w/2)
    data['total'] = data.rolling(window=w).sum()['logReturns'].shift(window_shift)
    data['std'] = data.rolling(window=w).std()['logReturns'].shift(window_shift)
    data['kurtosis'] = data.rolling(window=w).kurt()['logReturns'].shift(window_shift)
    data['mean'] = data.rolling(window=w).mean()['logReturns'].shift(window_shift)
    data['skew'] = data.rolling(window=w).skew()['logReturns'].shift(window_shift)



rolling_stats(21)

# sort pandas on highest/lowest return days worst days at top, best at bottom
sorted_data = data.sort_values(by='logReturns', ascending=True)
best_days = sorted_data.tail(50)
worst_days = sorted_data.head(50)




plt.figure(figsize=(15,15))
plt.suptitle("Best/Worst one day returns vs Probability Asymmetry in the month leading up to it")


plt.subplot(1, 2, 1)
plt.ylim(-1.0, 1.0)
plt.xlim(-.15, .15)
plt.scatter(best_days['logReturns'], best_days['skew'])
plt.xlabel('Best Log Returns')
plt.ylabel('Skew')

plt.subplot(1,2,2)
plt.ylim(-1.0, 1.0)
plt.xlim(-.15, .15)
plt.scatter(worst_days['logReturns'], worst_days['skew'])
plt.xlabel('Worst Log Returns')
plt.ylabel('Skew')

plt.savefig("RollingSkew.png")

plt.show()
