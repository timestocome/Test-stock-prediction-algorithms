

# http://github.com/timestocome
#
# let's take another look at the gain loss curves
# 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.mlab as mlab



###########################################################################
# data has been combined using LoadAndMatchDates.py
# raw data is from finance.yahoo.com
###########################################################################
data = pd.read_csv('StockDataWithVolume.csv', index_col='Date', parse_dates=True)


# convert to log scale
data['NASDAQ'] = np.log(data['NASDAQ'])
data['S&P'] = np.log(data['S&P'])
data['DJIA'] = np.log(data['DJIA'])


# add volatility
# >1 increase over yesterday
# <1 decrease over yesterday
data['NASDAQ_dx'] = data['NASDAQ'] / data['NASDAQ'].shift(1)
data['DJIA_dx'] = data['DJIA'] / data['DJIA'].shift(1)
data['S&P_dx'] = data['S&P'] / data['S&P'].shift(1)


data = data.dropna(axis=0)

# sanity check numbers
# print(data.head())


################################################################
# Let's see what kind of curves we have here.
#
# Most ML is done with Guassian Curves,
# most economic data turns out to be Zipf Distributions 
###############################################################

plt.figure(figsize=(12,12))
bins = 500

plt.subplot(311)
n, bins, patches = plt.hist(data['NASDAQ_dx'], bins, normed=1)

plt.ylabel('Probability')
plt.title('Histogram NASDAQ Daily Gains/Losses')
plt.grid(True)
plt.xlim(.98, 1.02)



plt.subplot(312)
n, bins, patches = plt.hist(data['DJIA_dx'], bins, normed=1)

plt.ylabel('Probability')
plt.title('Histogram DJIA Daily Gains/Losses')
plt.grid(True)
plt.xlim(.98, 1.02)


plt.subplot(313)
n, bins, patches = plt.hist(data['S&P_dx'], bins, normed=1)

plt.xlabel('Gains > 1, Losses < 1, No Change = 1')
plt.ylabel('Probability')
plt.title('Histogram S&P 500 Daily Gains/Losses')
plt.grid(True)
plt.xlim(.98, 1.02)

plt.savefig('histogram.png')
plt.show()

