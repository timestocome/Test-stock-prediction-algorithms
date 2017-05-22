# http://github.com/timestocome

# take a look at the differences in daily returns for recent bull and bear markets


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 


# pandas display options
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 25
pd.options.display.width = 1000

######################################################################
# data
########################################################################

# read in datafile created in LoadAndMatchDates.py
data = pd.read_csv('StockDataWithVolume.csv', index_col='Date', parse_dates=True)
features = [data.columns.values]


# create target --- let's try Nasdaq value 1 day change
data['returns'] = (data['NASDAQ'] - data['NASDAQ'].shift(1)) / data['NASDAQ']


# remove nan row from target creation
data = data.dropna()

'''
############################################################################
# plot returns on NASDAQ training data
#############################################################################

fig = plt.figure(figsize=(10,10))
plt.subplot(2,1,2)


plt.subplot(2,1,1)
plt.plot(data['returns'])
plt.title("Nasdaq daily returns")


# histogram of returns
plt.subplot(2,1,2)
plt.hist(data['returns'], bins=200)
plt.xlabel("Returns")
plt.ylabel("Probability")
plt.title("Histogram daily Nasdaq returns")
plt.grid(True)

# median
median_return = data['returns'].median()
l = plt.axvspan(median_return-0.0001, median_return+0.0001, color='red')

plt.show()
'''
#########################################################################
# split into bear and bull markets
##########################################################################

bull1_start = pd.to_datetime('01-01-1990')       # beginning of this dataset
bull1_end = pd.to_datetime('07-16-1990')

iraq_bear_start = pd.to_datetime('07-17-1990')
iraq_bear_end = pd.to_datetime('10-11-1990')

bull2_start = pd.to_datetime('10-12-1990')
bull2_end = pd.to_datetime('01-13-2000')

dotcom_bear_start = pd.to_datetime('01-14-2000')
dotcom_bear_end = pd.to_datetime('10-09-2002')

bull3_start = pd.to_datetime('10-10-2002')
bull3_end = pd.to_datetime('10-08-2007')

housing_bear_start = pd.to_datetime('10-09-2007')
housing_bear_end = pd.to_datetime('03-09-2009')

bull4_start = pd.to_datetime('03-10-2009')
bull4_end = pd.to_datetime('12-31-2016')    # end of this dataset



bull1 = data.loc[data.index <= bull1_end]
bear1 = data.loc[(data.index >= iraq_bear_start) & (data.index <= iraq_bear_end)]
bull2 = data.loc[(data.index >= bull2_start) & (data.index <= bull2_end)]
bear2 = data.loc[(data.index >= dotcom_bear_start) & (data.index <= dotcom_bear_end)]
bull3 = data.loc[(data.index >= bull3_start) & (data.index <= bull3_end)]
bear3 = data.loc[(data.index >= housing_bear_start) & (data.index <= housing_bear_end)]
bull4 = data.loc[data.index >= bull4_start]

####################################################################
# plot bear/bull markets - only the complete ones -
###################################################################
plt.figure(figsize=(16,16))
n_bins = 40
plt.suptitle("Returns for full bear/bull markets Jan 1990-Dec 2015")


plt.subplot(7,2,1)
plt.title("Jan 90-Jul 90 Bull")
plt.plot(bull1['returns'], color='green')
plt.ylim(-0.15, .15)
plt.xlim(pd.to_datetime('01-01-1990'), pd.to_datetime('12-31-2016'))


plt.subplot(7,2,2)
plt.title("Jan 90-Jul 90 Bull")
plt.hist(bull1['returns'], range=[-0.15, 0.15], bins=n_bins, color='green', normed=True)
plt.ylim(0, 50)
median_return = bull1['returns'].median()
l = plt.axvspan(median_return-0.001, median_return+0.001, color='yellow')


plt.subplot(7,2,3)
plt.title("July90-Oct 90")
plt.plot(bear1['returns'], color='red')
plt.ylim(-0.15, .15)
plt.xlim(pd.to_datetime('01-01-1990'), pd.to_datetime('12-31-2016'))

plt.subplot(7,2,4)
plt.title("July 90-Oct 90")
plt.hist(bear1['returns'], range=[-0.15, 0.15], bins=n_bins, color='red', normed=True)
plt.ylim(0, 50)
median_return = bear1['returns'].median()
l = plt.axvspan(median_return-0.001, median_return+0.001, color='yellow')


plt.subplot(7,2,5)
plt.title("Oct 90-Jan 00")
plt.plot(bull2['returns'], color='green')
plt.ylim(-0.15, .15)
plt.xlim(pd.to_datetime('01-01-1990'), pd.to_datetime('12-31-2016'))

plt.subplot(7,2,6)
plt.title("Oct 90-Jan 00")
plt.hist(bull2['returns'], range=[-0.15, 0.15], bins=n_bins, color='green', normed=True)
plt.ylim(0, 50)
median_return = bull2['returns'].median()
l = plt.axvspan(median_return-0.001, median_return+0.001, color='yellow')


plt.subplot(7,2,7)
plt.title("Jan 00-Oct 02")
plt.plot(bear2['returns'], color='red')
plt.ylim(-0.15, .15)
plt.xlim(pd.to_datetime('01-01-1990'), pd.to_datetime('12-31-2016'))

plt.subplot(7,2,8)
plt.title("Jan 00-Oct 02")
plt.hist(bear2['returns'], range=[-0.15, 0.15], bins=n_bins, color='red', normed=True)
plt.ylim(0, 50)
median_return = bear2['returns'].median()
l = plt.axvspan(median_return-0.001, median_return+0.001, color='yellow')


plt.subplot(7,2,9)
plt.title("Oct 02-Oct 07")
plt.plot(bull3['returns'], color='green')
plt.ylim(-0.15, .15)
plt.xlim(pd.to_datetime('01-01-1990'), pd.to_datetime('12-31-2016'))


plt.subplot(7,2,10)
plt.title("Oct 02-Oct 07")
plt.hist(bull3['returns'], range=[-0.15, 0.15], bins=n_bins, color='green', normed=True)
plt.ylim(0, 50)
median_return = bull3['returns'].median()
l = plt.axvspan(median_return-0.001, median_return+0.001, color='yellow')


plt.subplot(7,2,11)
plt.title("Oct 07-Mar 09")
plt.plot(bear3['returns'], color='red')
plt.ylim(-0.15, .15)
plt.xlim(pd.to_datetime('01-01-1990'), pd.to_datetime('12-31-2016'))


plt.subplot(7,2,12)
plt.title("Oct 07-Mar 09")
plt.hist(bear3['returns'], range=[-0.15, 0.15], bins=n_bins, color='red', normed=True)
plt.ylim(0, 50)
median_return = bear3['returns'].median()
l = plt.axvspan(median_return-0.001, median_return+0.001, color='yellow')


plt.subplot(7,2,13)
plt.title("Mar 09-Dec 15")
plt.plot(bull4['returns'], color='green')
plt.ylim(-0.15, .15)
plt.xlim(pd.to_datetime('01-01-1990'), pd.to_datetime('12-31-2016'))


plt.subplot(7,2,14)
plt.title("Mar 09-Dec 15")
plt.hist(bull4['returns'], range=[-0.15, 0.15], bins=n_bins, color='green', normed=True)
plt.ylim(0, 50)
median_return = bull4['returns'].median()
l = plt.axvspan(median_return-0.001, median_return+0.001, color='yellow')






plt.savefig("RecentBullBear.png")
plt.show()