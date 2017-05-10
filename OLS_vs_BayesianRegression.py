# http://github.com/timestocome

# Chpt 11-Portfolio Optimization in Python for Finance, O'Reilly
# I couldn't get the book example to put out anything useful
# trying http://scikit-learn.org/stable/auto_examples/linear_model/plot_bayesian_ridge.html
# use data 1990-2015 for training
# predict price 252 trading days, 1 year ahead
# out of the box classifiers Ordinary Least Squares, Bayes Regression return same results


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import BayesianRidge, LinearRegression
import matplotlib.pyplot as plt 



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
    #stock = stock.loc[stock.index < '12-31-2016']


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

'''
# compare indexes
(data / data.ix[0] * 100).plot(figsize=(12,12))
plt.title("Standarized Indexes 1990-2016")
plt.show()
'''

# predict next year's price
dow_jones['Future'] = dow_jones['Open'].shift(-252)

# drop Nan
dow_jones = dow_jones.dropna()

train = dow_jones.loc[dow_jones.index < '12-31-2015']
test = dow_jones.loc[dow_jones.index > '12-31-2015']




# prep for data for classifiers
x = train['Open'].as_matrix()
y = train['Future'].as_matrix()

x_test = test['Open'].as_matrix()
y_test = test['Future'].as_matrix()

# reshape into (row, column for sklearn)
x = x.reshape((len(x), 1))
y = y.reshape((len(y), 1))


x_test = x_test.reshape(len(x_test), 1)
y_test = y_test.reshape(len(y_test), 1)


# fit classifiers
ols = LinearRegression()
ols = ols.fit(x, y)
predict_ols = ols.predict(x_test)
score_ols = ols.score(x_test, y_test)

clf = BayesianRidge(compute_score=True)
clf = clf.fit(x, y)
predict_b = clf.predict(x_test)
score_b = clf.score(x_test, y_test)

print("Accuracy: OLS %lf, Bayes %lf" % (score_ols, score_b))


# plot results
plt.plot(y_test, 'r+', label="actual")
plt.plot(predict_ols, 'bx', label="ols")
plt.plot(predict_b, 'g1', label="bayesian")
plt.legend()
plt.title("Predict DJIA 1 year ahead ( 2016 )")
plt.show()