# http://github.com/timestocome


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# look for signature patterns before very high or low return days
# meh, the month before tends to have more extreme returns 
# and more of them are lower than average, (like an earthquake?)
# but there isn't a clear signature pattern


# read in nasdaq 
data = pd.read_csv('data/nasdaq.csv', parse_dates=True, index_col='Date')

# reverse dates
#data = data.iloc[::-1]


# keep only opening price
data = data[['Open']]
data['Open'] = pd.to_numeric(data['Open'], errors='coerce')

# log of price is used in most stock market calculations
data['LogOpen'] = np.log(data[['Open']]) 

# index change from prior day
data['dx'] = data['LogOpen'] - data['LogOpen'].shift(1)

# check every thing looks okay
# print(data)


# look for patterns in the daily change 

# redo dx as high gain 4, low gain 3, low loss 2, high loss 1
# use this to set the dividers between high and average return days
print(data.dx.mean(), data.dx.median(), data.dx.max(), data.dx.min())

def gains_losses(d):

    if d >= 0.05: return 4
    elif d > 0.00: return 3
    elif d <= -0.05: return 1
    else: return 2

data['GainLoss'] = data['dx'].apply(gains_losses)
print("Count of days with good, bad and average returns: \n", data['GainLoss'].value_counts())


# what happens in the n trading days before a very good or bad day?
n_trading_days = 21     # 5/week, 21/month, 63/quarter, 252/year

# add a row count column to make it easier to fetch data slices
i = np.arange(1, len(data) + 1)
data['i'] = i

# set up storage
lowReturns = []
highReturns = []
slightlyLowReturns = []
slightlyHighReturns = []

for idx, row in data.iterrows():
    if row.i > n_trading_days:

        start = int(row.i - n_trading_days)
        end = int(row.i)
        pattern = np.asarray(data.iloc[start:end, :]['dx'].values)

        if row.GainLoss == 1:       # very bad day 
            lowReturns.append(pattern)

        if row.GainLoss == 2:       
            slightlyLowReturns.append(pattern)

        if row.GainLoss == 3:       
            slightlyHighReturns.append(pattern)

        if row.GainLoss == 4:       # very good day
            highReturns.append(pattern)


# create np array columns = n_trading_days before high return day
high_returns = np.array(highReturns)
low_returns = np.array(lowReturns)
slightly_low_returns = np.array(slightlyLowReturns)
slightly_high_returns = np.array(slightlyHighReturns)


print(high_returns.shape)
print(slightly_high_returns.shape)
print(slightly_low_returns.shape)
print(low_returns.shape)

high_avg = high_returns.mean(axis=0)
low_avg = low_returns.mean(axis=0)
slightlyHigh_avg = np.nanmean(slightly_high_returns, axis=0)
slightlyLow_avg = np.nanmean(slightly_low_returns, axis=0)




for i in range(n_trading_days):
    print('%.5f, %.5f, %.5f, %.5f' %(high_avg[i], slightlyHigh_avg[i], slightlyLow_avg[i], low_avg[i]))




plt.figure(figsize=(16,12))
plt.title("21 day returns before a large gain/loss are mostly losses and larger than average")
plt.plot(high_avg, lw=3, label="very high returns")
plt.plot(slightlyHigh_avg, label="gains")
plt.plot(slightlyLow_avg, label="losses")
plt.plot(low_avg, lw=3, label='big losses')
plt.legend(loc='best')

plt.savefig("FindPatterns1.png")
plt.show()





# heat maps of best worst trading days
plt.figure(figsize=(16, 16))
plt.suptitle("Patterns before extreme high/low return trading days")

plt.subplot(121)
map_best = plt.imshow(high_returns, cmap=plt.cm.Spectral, interpolation='nearest')
plt.title("21 days leading to highest return days")

plt.subplot(122)
map_worst = plt.imshow(low_returns, cmap=plt.cm.Spectral, interpolation='nearest')
plt.title("21 days leading to lowest return days")


cbar = plt.colorbar(map_best)
cbar.ax.set_yticklabels(['High loss', '', 'low loss', '', 'low gain', '', 'High Gain', ''])
cbar.set_label("Colors")

plt.savefig("FindPatterns2.png")

plt.show()




'''
plt.figure(figsize=(18,14))
plt.plot(data['Open'], label='Nasdaq')
plt.plot(data['LogOpen'] * 100, label='Scaled Log')
plt.title("Nasdaq Composite Index")
plt.legend(loc='best')
plt.show()
'''