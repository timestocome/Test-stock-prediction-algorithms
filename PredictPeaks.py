# http://github.com/timestocome

# take a look at the differences in daily returns for recent bull and bear markets
# plot daily returns for a year, check plot agains year's return 
# power rule histograms rule in good years, flatter ones in bear markets
# looks like the histogram flattens as the market peaks, might be a leading indicator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


######################################################################
# data
########################################################################

# read in datafile created in LoadAndMatchDates.py
data = pd.read_csv('StockDataWithVolume.csv', index_col='Date', parse_dates=True)
features = [data.columns.values]



# switch to log data
data['logDJIA'] = np.log(data['DJIA'])
data['logNASDAQ'] = np.log(data['NASDAQ'])
data['logS&P'] = np.log(data['S&P'])
data['logRussell'] = np.log(data['Russell 2000'])


# log returns
data['logReturnsNASDAQ'] = data['logNASDAQ'] - data['logNASDAQ'].shift(1)
data['logReturnsDJIA'] = data['logDJIA'] - data['logDJIA'].shift(1)
data['logReturnsS&P'] = data['logS&P'] - data['logS&P'].shift(1)
data['logReturnsRussell'] = data['logRussell'] - data['logRussell'].shift(1)


# remove nan row from target creation
data = data.dropna()

'''
bins = [-0.16, -0.14, -0.12, -0.10, -0.08, -0.06, -0.04,  -0.02, 0.0, 0.02, 0.04, 0.06, 0.07, 0.08, 0.10, 0.12, 0.14, 0.16]
def plot_histogram(d):
    global year
    n, b, _ = plt.hist(d, bins=bins)
    return (n, b)
h_plots = data['logReturnsNASDAQ'].groupby([data.index.year]).apply(plot_histogram)

'''

yr_sum = data.rolling(window=252, center=True).sum()['logReturnsNASDAQ']
yr_std = data.rolling(window=252, center=True).std()['logReturnsNASDAQ']
yr_median = data.rolling(window=252, center=True).median()['logReturnsNASDAQ']
yr_kurtosis = data.rolling(window=252, center=True).kurt()['logReturnsNASDAQ']


plt.figure(figsize=(12,12))
plt.plot(data['logNASDAQ'], label='NASDAQ')
plt.plot(data['logReturnsNASDAQ'], label='Log Returns')
plt.plot(yr_sum, label='Yearly sum')
plt.plot(yr_std * 100., label='Yearly std')
plt.plot(yr_median, label='Yearly median')
plt.plot(yr_kurtosis + 5, label='Yearly kurtosis')

plt.legend(loc='best')

plt.savefig("Std_Kurtosis_MarketPeaks.png")
plt.show()