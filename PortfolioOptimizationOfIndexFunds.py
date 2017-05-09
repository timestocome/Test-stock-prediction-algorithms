# http://github.com/timestocome

# Chpt 11-Portfolio Optimization in Python for Finance, O'Reilly

# Not sure what to say about this stuff. It's a good review of how
# things used to be done and from what I've seen is still in use by
# some mutual funds which is why you should buy index funds or do your 
# own investing.


import pandas as pd
import numpy as np
import numpy.random as npr
import scipy.stats as scs
import scipy.optimize as sco
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import matplotlib as matplotlib





# read in file
def read_data(file_name):

    stock = pd.read_csv(file_name, parse_dates=True, index_col=0)        # 31747 days of data 
    n_samples = len(stock)
    
    # ditch samples with NAN values
    stock = stock.dropna(axis=0)

    # flip order from newest to oldest to oldest to newest
    stock = stock.iloc[::-1]

    # trim data
    stock = stock[['Open']]

    # trim dates
    stock = stock.loc[stock.index > '01-01-1990']
    stock = stock.loc[stock.index < '12-31-2016']


    # all stock is needed to walk back dates for testing hold out data
    return stock


#############################################################################################
# load and combine stock indexes 


dow_jones = read_data('data/djia.csv')
print("Loaded DJIA", len(dow_jones))

s_p = read_data('data/S&P.csv')
print("Loaded S&P", len(s_p))

russell_2000 = read_data('data/Russell2000.csv')
print("Loaded Russell", len(russell_2000))

nasdaq = read_data('data/nasdaq.csv')
print("Loaded NASDAQ", len(nasdaq))

# combine stock indexes into one dataframe
data = pd.concat([dow_jones['Open'], s_p['Open'], russell_2000['Open'], nasdaq['Open']], axis=1, keys=['dow_jones', 'S&P', 'russell_2000', 'nasdaq'])

# compare indexes
(data / data.ix[0] * 100).plot(figsize=(12,12))
plt.title("Standarized Indexes 1990-2016")
plt.show()

########################################################################################
# log returns
# continuous rate e^(rt), using log normalizes returns
# shows differences in returns on the 4 indexes 

def print_statistics(array):

    sta = scs.describe(array)
    print("%14s %15s" % ('statistic', 'value'))
    print(30 * '-')
    print("%14s %15.5f" % ('size', sta[0]))
    print("%14s %15.5f" % ('min', sta[1][0]))
    print("%14s %15.5f" % ('max', sta[1][1]))
    print("%14s %15.5f" % ('mean', sta[2] ))
    print("%14s %15.5f" % ('std', np.sqrt(sta[3])))
    print("%14s %15.5f" % ('skew', sta[4]))
    print("%14s %15.5f" % ('kutosis', sta[5]))
  


log_returns = np.log(data / data.shift(1))

# compare indexes
annualized_returns = log_returns.mean() * 252
print("Annualized returns 1990-2016")
print(annualized_returns)

# plot
log_returns.hist(bins=50, figsize=(12,12))
plt.suptitle("Histogram of log returns")
plt.show()


# print to screen for user
stocks = data.columns.values
for stock in stocks:
    print(30 * '-')
    print ("\n Results for index %s" % stock)
    log_data = np.array(log_returns[stock].dropna())
    print_statistics(log_data)



###############################################################################
# Quantile-quantile - calculate and plot
# shows fat tail distribution in all 4 of these indexes
# note curves on both ends

sm.qqplot(log_returns['dow_jones'].dropna(), line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
plt.title('Quantiles Dow Jones')
plt.show()


sm.qqplot(log_returns['S&P'].dropna(), line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
plt.title('Quantiles S&P')
plt.show()


sm.qqplot(log_returns['russell_2000'].dropna(), line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
plt.title('Quantiles Russell 2000')
plt.show()


sm.qqplot(log_returns['nasdaq'].dropna(), line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
plt.title('Quantiles Nasdaq')
plt.show()




#########################################################################################
# portfolio balancing 1-1-1990, 12-31-2016
# 252 is the estimated trading days in one year


# try equal investments
covariance_indexes = log_returns.cov() * 252
print(30 * '-')
print("Covariance matrix for indexes")
print(covariance_indexes)
print(30 * '-')


# invest 25% of your money in each of the 4 indexes
weights = np.asarray([.25, .25, .25, .25])

portfolio_return = np.sum(log_returns.mean() * weights) * 252
print(30 * '-')
print("return on equal weight portfolio", portfolio_return)

portfolio_variance = np.dot(weights.T, np.dot(log_returns.cov() * 252, weights))
print("expected portfolio variance on returns", portfolio_variance)
print("expected returns %.2lf to %.2lf" %(portfolio_return - portfolio_variance, portfolio_return + portfolio_variance))
print(30 * '-')

##############################################################################
# use optimization to get best weighting for portfolio


# weights must add up to 1 ( 100% of money invested)
noa = len(stocks)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for x in range(noa))


def statistics(weights):
    
    weights = np.array(weights)
    p_return = np.sum(log_returns.mean() * weights) * 252
    p_volatility = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))

    return (np.array([p_return, p_volatility, p_return/p_volatility]))



# maximize mean of returns
# http://www.investopedia.com/terms/s/sharperatio.asp
# Sharpe ratio === (expected_return - current_savings_rate ) / (std_of_portfolio)
def min_func_sharpe(weights):
    return -statistics(weights)[2]

optimal_values = sco.minimize(min_func_sharpe, noa * [1./noa,], method='SLSQP', bounds=bounds, constraints=constraints)
print("Optimal value portfolio")
print(optimal_values)

print("Optimal weights for maximum return", optimal_values['x'].round(3))
print("Expected return, volatility, Sharpe ratio", statistics(optimal_values['x'].round(3)))
print(30 * '-')




# minimize variance
def min_func_variance(weights):
    return statistics(weights)[1] **2

optimal_variance = sco.minimize(min_func_variance, noa * [1. / noa,], method='SLSQP', bounds=bounds, constraints=constraints)
print("Optimal variance portfolio")
print(optimal_variance)

print("Optimal weights for low variance", optimal_variance['x'].round(3))
print("Expected return, volatility, Sharpe ratio", statistics(optimal_variance['x'].round(3)))
print(30 * '-')


