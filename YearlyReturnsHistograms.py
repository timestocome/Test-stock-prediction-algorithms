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


bins = [-0.16, -0.14, -0.12, -0.10, -0.08, -0.06, -0.04,  -0.02, 0.0, 0.02, 0.04, 0.06, 0.07, 0.08, 0.10, 0.12, 0.14, 0.16]
def plot_histogram(d):
    global year
    n, b, _ = plt.hist(d, bins=bins)
    return (n, b)


plot_histogram(data['logReturnsNASDAQ'])
h_plots = data['logReturnsNASDAQ'].groupby([data.index.year]).apply(plot_histogram)


plots = []
y = 1990
r = 3
c = 9
plot_n = 1
plt.figure(figsize=(30, 12))
for i, p in h_plots.iteritems():
    plt.subplot(r, c, plot_n)
    n = p[0]
    bins = p[1]
    plt.bar(bins[:-1], n, width=.02)
    plt.xlim(min(bins), max(bins))
    start_date = '01/01/%d' % y
    end_date = '12/31/%d' % y
    yearly_returns = data.loc[(data.index >= start_date) & (data.index <= end_date) ]
    plt.title("%d  LogNet: %f" % (y, yearly_returns['logReturnsNASDAQ'].sum()))
    y += 1
    plot_n += 1

plt.savefig('BullBearHistograms.png')
plt.show()    

