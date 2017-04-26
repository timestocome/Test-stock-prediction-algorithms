# http://github.com/timestocome



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os

pd.options.display.max_rows = 100
pd.options.display.max_columns = 25
pd.options.display.width = 1000

# http://faculty.chicagobooth.edu/workshops/finance/pdf/Shleiferbff.pdf
# Paper looks at acceleration, number of new businesses and volality

# What can be told just using acceleration on the daily price?
# seems to be a pattern between absolute value of acceleration and bull/bear markets
# Acceration seems to peak just before market bottoms out, 
# not as clear indicator for market topping out.




def movement(file_name, dailyPrice):

    # read in file
    stock = pd.read_csv(file_name, parse_dates=True, index_col=0)        # 31747 days of data 
    n_samples = len(stock)


    # daily change in stock (derivative)
    stock['dx'] = stock[dailyPrice] - stock[dailyPrice].shift(1)      # points
    stock['percent_dx'] = stock['dx'] / stock[dailyPrice]         # percent

    # 2nd derviative
    stock['d2x'] = stock['dx'] - stock['dx'].shift(1) 
    stock['percent_d2x'] = stock['d2x'] / stock[dailyPrice]


    # abs value of dx, d2x, 
    w = 92
    stock['dx_abs'] = stock['dx'].abs()
    stock['d2x_abs'] = stock['d2x'].abs()
    stock['dx_abs_ma'] = stock['dx_abs'].rolling(window=w).mean()
    stock['d2x_abs_ma'] = stock['d2x_abs'].rolling(window=w).mean()
    stock['stock_ma'] = stock[dailyPrice].rolling(window=w).mean()
    stock['dx_ma'] = stock['dx'].rolling(window=w).mean()
    stock['d2x_ma'] = stock['d2x'].rolling(window=w).mean()

    # momentum of price
    stock['momentum'] = (stock[dailyPrice] * stock['dx'] ).rolling(window=w).mean()


    # write to disk
    print(os.path.basename(file_name))
    f_name = 'data/vma_' + os.path.basename(file_name)
    stock.to_csv(f_name)


    correlations = stock.corr()

    # plot 
    recent = stock.loc[stock.index > '01-01-1980']

    fig = plt.figure(figsize=(18, 10))  # in inches
    ax = fig.add_subplot(111)
    ax.xaxis.grid(True, which='major')


    title = '%s Daily Stock, Abs Acceleration, Momentum' % ( file_name)
    fig.suptitle(title)

    ax.plot(recent['stock_ma'], color='navy', linewidth=3, label=dailyPrice)
    # scale to make plots more readable
    ax.plot(recent['d2x_abs_ma'] * 20., c='magenta', linewidth=1, label='Acceleration (abs)')
    ax.plot(recent['momentum'] / stock[dailyPrice].mean() + stock[dailyPrice].mean(), c='green', linewidth=1, label='Momentum')
    legend = ax.legend(loc='upper left')



    return plt, correlations




file_name = 'data/djia.csv'
dailyPrice = 'Close'
dja, dja_corr = movement(file_name, dailyPrice)

file_name = 'data/nasdaq.csv'
dailyPrice = 'Close'
nasdaq, nasdaq_corr = movement(file_name, dailyPrice)

file_name = 'data/S&P.csv'
dailyPrice = 'Close'
sp, sp_corr = movement(file_name, dailyPrice)

file_name = 'data/Russell2000.csv'
dailyPrice = 'Close'
russell, russell_corr = movement(file_name, dailyPrice)

#plt.show()
print("Correlations")
print('DJIA')
print(dja_corr['Close'])

print("NASDAQ")
print(nasdaq_corr['Close'])

print("S&P 500")
print(sp_corr['Close'])

print("Russell 200")
print(russell_corr['Close'])


'''
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

'''
