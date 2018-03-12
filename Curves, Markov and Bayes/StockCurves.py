

# http://github.com/timestocome
#
# Data is from http://finance.yahoo.com
# 
#
# let's take another look at the gain loss curves
# 
# Whoa, baby look at that BitCoin Volatility distribution
# what does it mean, idk?
# there is the obvious 
# it's a highly volitile market prone to sudden changes
# 
# might also mean more
# Guassian distributions depend on lots of investors acting independently
# a flat Platykurtic might mean that's not the case?
# But that's probably obvious too, 
# I'm still looking for more information



import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


#############################################################################
# compute stats
#############################################################################
pd.set_option('display.max_rows', 5000)



# http://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm
def statistics(d):
    
    d = d.replace([np.inf, -np.inf], np.nan)
    
    m = np.mean(d)
    v = np.var(d)
    std = np.std(d)
    n = len(d)
    
    # Fisher-Pearson coffeicient of skewness
    g1 = np.sum((d - m)**3 / n) / (std**3)
    #print('g1', g1)
    
    # size adjusted skew
    G1 = (np.sqrt(n*(n-1)))/(n-2) * g1
    #print('G1', G1)
    
    
    # kurtosis
    kurt = np.sum((d - m)**4 / n) / (std**4)
    
    # exessive kurtosis
    e_kurt = kurt - 3
    
    '''
    print('----------------------------')
    print('mean', m)
    print('var', v)
    print('std', std)
    print('skew', G1, skew(d))
    print('kurtosis', kurt, kurtosis(d))
    print('e kurt', e_kurt)
    print('------------------------------')
    '''
    
    return m, v, std, G1, e_kurt
   


###########################################################################
# data has been combined using LoadAndMatchDates.py
# raw data is from finance.yahoo.com
###########################################################################
data = pd.read_csv('StockDataWithVolume.csv', index_col='Date', parse_dates=True)


# convert to log scale
data['NASDAQ'] = np.log(data['NASDAQ'])
data['S&P'] = np.log(data['S&P'])
data['DJIA'] = np.log(data['DJIA'])
data['Russell'] = np.log(data['Russell'])
data['BTC'] = np.log(data['BTC'])


# add volatility
# >1 increase over yesterday
# <1 decrease over yesterday
data['NASDAQ_dx'] = data['NASDAQ'] / data['NASDAQ'].shift(1)
data['DJIA_dx'] = data['DJIA'] / data['DJIA'].shift(1)
data['S&P_dx'] = data['S&P'] / data['S&P'].shift(1)
data['BTC_dx'] = data['BTC'] / data['BTC'].shift(1)
data['Russell_dx'] = data['Russell'] / data['Russell'].shift(1)

data = data.dropna(axis=0)
data = data.replace([np.inf, -np.inf], np.nan)


# sanity check numbers
#print(data.head())

statistics(data['NASDAQ_dx'])
statistics(data['DJIA_dx'])
statistics(data['Russell_dx'])
statistics(data['S&P_dx'])
statistics(data['BTC_dx'])




################################################################
# Let's see what kind of curves we have here.
#
# Most ML is done with Guassian Curves,
# most economic data turns out to be Zipf Distributions 
###############################################################

plt.figure(figsize=(12,22))
bins = 100


plt.subplot(511)
n, bins, patches = plt.hist(data['NASDAQ_dx'], bins, normed=1)
kurt = kurtosis(data['NASDAQ_dx'])
skw = skew(data['NASDAQ_dx'])

mu, sigma = norm.fit(data['NASDAQ_dx'])
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)

t = 'Nasdaq mean ' + str(mu) + ',  std ' + str(sigma) +  ',  Kurtosis ' + str(kurt) + ',  Skew ' + str(skw)

plt.ylabel('Probability')
plt.title(t)
plt.grid(True)
plt.xlim(.98, 1.02)
plt.ylim(0, 800)



plt.subplot(512)
n, bins, patches = plt.hist(data['DJIA_dx'], bins, normed=1)
kurt = kurtosis(data['DJIA_dx'])
skw = skew(data['DJIA_dx'])

mu, sigma = norm.fit(data['DJIA_dx'])
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)

t = 'DJIA mean ' + str(mu) + ',  std ' + str(sigma) + ', Kurtosis ' + str(kurt) + ', Skew ' + str(skw)

plt.ylabel('Probability')
plt.title(t)
plt.grid(True)
plt.xlim(.98, 1.02)
plt.ylim(0, 800)


plt.subplot(513)
n, bins, patches = plt.hist(data['S&P_dx'], bins, normed=1)
kurt = kurtosis(data['S&P_dx'])
skw = skew(data['S&P_dx'])

mu, sigma = norm.fit(data['S&P_dx'])
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
t = 'S&P mean ' + str(mu) + ',  std ' + str(sigma) + ', Kurtosis ' + str(kurt) + ', Skew ' + str(skw)


plt.ylabel('Probability')
plt.title(t)
plt.grid(True)
plt.xlim(.98, 1.02)
plt.ylim(0, 800)



plt.subplot(514)
n, bins, patches = plt.hist(data['Russell_dx'], bins, normed=1)
kurt = kurtosis(data['Russell_dx'])
skw = skew(data['Russell_dx'])

mu, sigma = norm.fit(data['Russell_dx'])
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
t = 'Russell mean ' + str(mu) + ',  std ' + str(sigma) + ', Kurtosis ' + str(kurt) + ', Skew ' + str(skw)


plt.ylabel('Probability')
plt.title(t)
plt.grid(True)
plt.xlim(.98, 1.02)
plt.ylim(0, 800)



# stats package chokes on BitCoin stats
plt.subplot(515)
n, bins, patches = plt.hist(data['BTC_dx'], bins, normed=1)

m, var, std, skw, kurt = statistics(data['BTC_dx'])

#mu, sigma = norm.fit(data['BTC_dx'])
y = mlab.normpdf(bins, m, std)
l = plt.plot(bins, y, 'r--', linewidth=4)
t = 'BTC mean ' + str(m) + ',  std ' + str(std) + ', Kurtosis ' + str(kurt) + ', Skew ' + str(skw)




plt.ylabel('Probability')
plt.title(t)
plt.grid(True)
plt.xlim(.98, 1.02)
plt.ylim(0, 800)
plt.xlabel('Gains > 1, Losses < 1, No Change = 1')


plt.savefig('histogram.png')
plt.show()

