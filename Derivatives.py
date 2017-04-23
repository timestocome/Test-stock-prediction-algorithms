# http://github.com/timestocome



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import dates



# http://faculty.chicagobooth.edu/workshops/finance/pdf/Shleiferbff.pdf
# Paper looks at acceleration, number of new businesses and volality

# What can be told just using acceleration?
# seems to be a pattern between absolute value of acceleration and bull/bear markets
# Acceration seems to peak just before market bottoms out, 
# not as clear indicator for market topping out.

##########################################################################
# data is from  https://www.measuringworth.com/datasets/stock/index.php
##########################################################################
# read in stock


#file_name = 'DJA.csv'
#valuation = 'DJIA'

#file_name = 'nasdaq.csv'
#valuation = 'Close'

#file_name = 'S&P.csv'
#valuation = 'Close'


file_name = 'Russell2000.csv'
valuation = 'Close'





stock = pd.read_csv(file_name, parse_dates=True, index_col=0)        # 31747 days of data 
n_samples = len(stock)


# daily change in stock (derivative)
stock['dx'] = stock[valuation] - stock[valuation].shift(1)      # points
stock['percent_dx'] = stock['dx'] / stock[valuation]         # percent

# 2nd derviative
stock['d2x'] = stock['dx'] - stock['dx'].shift(1) 
stock['percent_d2x'] = stock['d2x'] / stock[valuation]

# scale djia
min_djia = stock[valuation].min()
max_djia = stock[valuation].max()
stock['Scaled_DJIA'] = (stock[valuation] - min_djia ) / ( max_djia - min_djia )


# abs value of dx
w = 92
stock['dx_abs'] = stock['dx'].abs()
stock['d2x_abs'] = stock['d2x'].abs()
stock['dx_abs_ma'] = stock['dx_abs'].rolling(window=w).mean()
stock['d2x_abs_ma'] = stock['d2x_abs'].rolling(window=w).mean()
stock['stock_ma'] = stock[valuation].rolling(window=w).mean()
stock['dx_ma'] = stock['dx'].rolling(window=w).mean()
stock['d2x_ma'] = stock['d2x'].rolling(window=w).mean()




# plot 
recent = stock.loc[stock.index > '01-01-1980']

fig = plt.figure(figsize=(18, 10))  # in inches
ax = fig.add_subplot(111)
ax.xaxis.grid(True, which='major')


title = '%s quarterly moving average vs abs acceleration moving average' % ( file_name)
fig.suptitle(title)

ax.plot(recent['stock_ma'], color='navy', linewidth=3, label=valuation)
ax.plot(recent['d2x_abs_ma'] * 20., c='magenta', linewidth=1, label='Acceleration (abs)')
#ax.plot(recent['dx_abs_ma'] * 20. + 7500, c='green', linewidth=1, label='Velocity (abs)')
legend = ax.legend(loc='upper left')

plt.show()



#############################################################################
# peak gain/losses in a day
#############################################################################
n_peaks = 100

top_gains = stock.nlargest(n_peaks, 'percent_dx')
top_gains = top_gains.sort_index()

top_losses = stock.nsmallest(n_peaks, 'percent_dx')
top_losses = top_losses.sort_index()

print("Top ", n_peaks)
print(top_gains[['percent_dx']])

print('-------------------------------------')
print("Bottom ", n_peaks)
print(top_losses[['percent_dx']])


