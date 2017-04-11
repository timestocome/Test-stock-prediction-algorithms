# http://github.com/timestocome



# http://pandas.pydata.org 
import pandas as pd

pd.options.display.max_rows = 100
pd.options.display.max_columns = 25
pd.options.display.width = 1000

import numpy as np
import matplotlib.pyplot as plt 



##########################################################################
# data is from  https://www.measuringworth.com/datasets/DJA/index.php
##########################################################################
# read in DJA
dja = pd.read_csv('cleaned_dja.csv',  index_col=0)        # 31747 days of data 
n_samples = len(dja)

# print("Columns")
# print(dja.columns.values)



###########################################################################
# calculate moving average 
###########################################################################

days_moving_average = 20

# moving averages
dja['MA'] = dja['Scaled_DJIA'].rolling(window=days_moving_average, center=False).mean()




#############################################################################
# peak gain/losses in a day
#############################################################################
n_peaks = 50

top_gains = dja.nlargest(n_peaks, 'percent_dx')
top_gains['dow'] = pd.DatetimeIndex(top_gains['Date']).dayofweek

top_losses = dja.nsmallest(n_peaks, 'percent_dx')
top_losses['dow'] = pd.DatetimeIndex(top_losses['Date']).dayofweek

print("Top ", n_peaks)
print(top_gains[['Date','dow', 'percent_dx']])

print('-------------------------------------')
print("Bottom ", n_peaks)
print(top_losses[['Date', 'dow', 'percent_dx']])




##################################################################################
# plot
#################################################################################

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

    
mg = pd.DatetimeIndex(top_gains['Date']).month
dg = pd.DatetimeIndex(top_gains['Date']).dayofweek
g = top_gains['percent_dx'] * 20000.


ml = pd.DatetimeIndex(top_losses['Date']).month
dl = pd.DatetimeIndex(top_losses['Date']).dayofweek
l = top_losses['percent_dx'] * -1. * 20000.


plt.figure(figsize=(18, 7))

# the market was open Sats in the 1930s
plt.yticks([0,1,2,3,4,5], days)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12], months)
plt.title("50 Best and worst days for stocks 1900-2017")

plt.scatter(mg, dg, s=g, c='darkgreen', alpha=0.4)
plt.scatter(ml + 0.1, dl + 0.1, s=l, c='maroon', alpha=0.6)

plt.savefig("50_best_worst_trading_days.png")

plt.show()


