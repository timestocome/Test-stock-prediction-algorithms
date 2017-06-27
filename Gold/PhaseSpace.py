
# http://github.com/timestocome

# look at phase space plots for gold prices
# see if any strange attractors appear
# US stop using gold standard 1971


# During the gold standard there is increased volatility during wars
# some volatility during wars after, but it's buried in speculation

# interesting gaps in the gold prices - attractor / repeller prices? 
# more digging is needed. 


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 





######################################################################
# load data
########################################################################
# read in gold file
data = pd.read_csv('data/Gold_all.csv', parse_dates=True, index_col=0)
data = data[['Open']]
data['Open'] = pd.to_numeric(data['Open'], errors='coerce')
data['Volatility'] = data['Open'] - data['Open'].shift(1)
data = data.dropna()

gold_standard = data.loc[data.index < '01-01-1971']
gold = data.loc[data.index > '01-01-1971']


print(len(gold_standard), len(gold))
########################################################################


plt.figure(figsize=(18, 15))
plt.subplot(3,1,1)
plt.plot(gold_standard['Open'], label='Value')
plt.plot(gold_standard['Volatility'] * 10, label='Volatility')
plt.legend(loc='best')
plt.title('Gold while US on gold standard')

plt.subplot(3,1,2)
plt.scatter(gold_standard['Open'], gold_standard['Volatility'])
plt.xlabel('Value')
plt.ylabel('Volatility')

plt.subplot(3,1,3)
n_bins = 100
plt.hist(gold_standard['Open'], n_bins, normed=1, histtype='bar')
plt.savefig("Gold_duringGoldStandard.png")

#----------------------------------------------------

plt.figure(figsize=(18, 15))
plt.subplot(3,1,1)
plt.plot(gold['Open'], label='Value')
plt.plot(gold['Volatility'] * 10, label='Volatility')
plt.legend(loc='best')
plt.title('Gold')

plt.subplot(3,1,2)
plt.scatter(gold['Open'], gold['Volatility'])
plt.xlim(0, 2000)
plt.xlabel('Value')
plt.ylabel('Volatility')

plt.subplot(3,1,3)
n_bins = 100
plt.xlim(0, 2000)
plt.hist(gold['Open'], n_bins, normed=1, histtype='bar')

plt.savefig("Gold_offGoldStandard.png")
plt.show()

