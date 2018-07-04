
# http://github.com/timestocome

# Level data so series is stationary in time
# take log of data
# save it to use in deconstructing signal to find anomolies
# Using finance.yahoo.com Nasdaq, S&P, DJI 1985 - date (Nov 29 2017)
# 
# https://blog.statsbot.co/time-series-anomaly-detection-algorithms-1cef5519aef2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


# pandas display options
pd.options.display.max_rows = 10000
pd.options.display.max_columns = 25
pd.options.display.width = 1000







######################################################################
# plot dataframe
########################################################################
def plot_dataframe(d, t):


    plt.figure(figsize=(18,18))
    plt.plot(d['NASDAQ'], label='NASDAQ')
    plt.plot(d['S&P'], label='S&P')
    plt.plot(d['DJIA'], label='DJIA')
    plt.plot(d['BTC'], label='BTC')
    plt.plot(d['Russell'], label='Russell')
    plt.title(t)
    plt.legend(loc='best')
    plt.show()


######################################################################
# data
########################################################################
# read in datafile created in LoadAndMatchDates.py
data = pd.read_csv('StockDataWithVolume.csv', index_col='Date', parse_dates=True)
features = ['DJIA', 'S&P', 'NASDAQ', 'Russell', 'BTC']


# fill in a couple NaN
#data.dropna()
data = data.fillna(method='ffill')



#########################################################################################
# level the series out, time series calculations all assume signal is stationary in time
########################################################################################

# pandas removed ols package !#&^*@$
# need y intercept, b
# and slope, m
# y = mx + b
# using simplest case possible
#
# how to get x, y just in case you want to put this into an ordinary least squares package
# for better slope/intercept numbers
# This is close enough for proof of concept
#
# x = list(range(1, len(data)))
# y = data



# not really ols, but close enough
def ols(data):
    m = (data[-1] - data[0]) / len(data)
    b = data[0]
    
    print(data[-1], data[0], (data[-1] - data[0]))
    print(m, b)
    
    
    print('-----------------------')

    return m, b
    



# add a time step 
steps = np.asarray(range(1, len(data)+1))
steps.reshape(1, -1)
data['step'] = steps



# NASDAQ
data['log NASDAQ'] = np.log(data['NASDAQ'])
m, b = ols(data['log NASDAQ'])
data['leveled log Nasdaq'] = data['log NASDAQ'] - (b + data['step'] * m)



# S&P
data['log S&P'] = np.log(data['S&P'])
m, b = ols(data['log S&P'])
data['leveled log S&P'] = data['log S&P'] - (b + data['step'] * m)



# DJIA
data['log DJIA'] = np.log(data['DJIA'])
m, b = ols(data['log DJIA'])
data['leveled log DJIA'] = data['log DJIA'] - (b + data['step'] *  m)


# BTC
data['log BTC'] = np.log(data['BTC'])
m, b = ols(data['log BTC'])
data['leveled log BTC'] = data['log BTC'] - (b + data['step'] * m)


# Russell
data['log Russell'] = np.log(data['Russell'])
m, b = ols(data['log Russell'])
data['leveled log Russell'] = data['log Russell'] - (b + data['step'] * m)




#print(data.columns.values)
data = data[['leveled log Nasdaq','leveled log S&P', 'leveled log DJIA', 'leveled log Russell', 'leveled log BTC']]


# save data
data.to_csv('LeveledLogStockData.csv')




# plot to make sure things look ok
plt.figure(figsize=(12,12))
plt.plot(data['leveled log Nasdaq'], label='NASDAQ')
plt.plot(data['leveled log S&P'], label='S&P')
plt.plot(data['leveled log DJIA'], label='DJIA')
plt.plot(data['leveled log BTC'], label='BTC')
plt.plot(data['leveled log Russell'], label='Russell')


plt.legend(loc='best')
plt.show()

