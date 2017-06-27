# http://github.com/timestocome


# how well you do depends on which phase of the market you 
# start on - Also you need to invest a large amount to cover 
# commissions 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




# variables to adjust
short_ma = 42      # ~ 21 days per trading month
long_ma = 252       # ~ 252 trading days per year
threshold = 50      # point difference
start_year = 1985


# file names for data
djia_file =  'data/djia.csv'
nasdaq_file = 'data/nasdaq.csv'
sp_file = 'data/S&P.csv'
russell_file = 'data/Russell2000.csv'
gold_file = 'data/GOLD.csv'



# read in file
def read_data(file_name):

    stock = pd.read_csv(file_name, parse_dates=True, index_col=0)        # 31747 days of data 
    n_samples = len(stock)

    # flip order from newest to oldest to oldest to newest
    stock = stock.iloc[::-1]

    return stock


# read in data
stock = read_data(sp_file)

# moving average
stock['short'] = np.round(stock['Close'].rolling(window=short_ma).mean(), 2)
stock['long'] = np.round(stock['Close'].rolling(window=long_ma).mean(), 2)


# plot
#stock[['Close', 'short', 'long']].plot(grid=True)
#plt.show()


###########################################################################
# test buy long, wait, sell short strategies
###########################################################################
stock['diff'] = stock['short'] - stock['long']

stock['plan'] = np.where(stock['diff'] > threshold, 1, 0)
stock['plan'] = np.where(stock['diff'] < -threshold, -1, stock['plan'])


# ditch old data
stock["Year"] = pd.DatetimeIndex(stock.index).year
stock = stock[stock['Year'] > start_year]


# plot plan
#stock['plan'].plot(linewidth=2)
#plt.ylim([-1.1, 1.1])
#plt.show()


# test plan 
# make market when invested (+/-1), make 0 when sitting on cash
# buy in when short term is more than threshold above long
# sell when long is more than threshold above short


# use returns instead of prices to normalize data
# use log to simplify the math and avoid skew ( http://www.dcfnerds.com/94/arithmetic-vs-logarithmic-rates-of-return/)
#stock['returns'] = np.log(stock['Close'].shift(-1) / stock['Close'])        # daily difference in price
stock['returns'] = np.log(stock['Close'] / stock['Close'].shift(1))        # daily difference in price
stock['strategy'] = stock['plan'].shift(1) * stock['returns']

# don't forget the commissions 
stock['fees'] = np.where(stock['strategy'] == 0, 0, 1)
stock['fees'] = stock['fees'] * 7.


stock[['returns', 'strategy']].cumsum().apply(np.exp).plot()
plt.title("S&P Returns vs Market Moving Averages 42/252 Fees: %.2lf " %(stock['fees'].sum()))
plt.show()


