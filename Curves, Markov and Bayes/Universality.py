#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# http://github.com/timestocome


# references
# 
# http://www2.math.uu.se/~takis/
# https://www.quantamagazine.org/in-mysterious-pattern-math-and-nature-converge-20130205/


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# read in saved file
###############################################################################
data = pd.read_csv('StockDataWithVolume.csv', index_col='Date', parse_dates=True)
n_rows = len(data)


# compute and save volatility
djia_v = data['DJIA'] / data['DJIA'].shift(1)
nasdaq_v = data['NASDAQ'] / data['NASDAQ'].shift(1)
sp_v = data['S&P'] / data['S&P'].shift(1)
russell_v = data['Russell'] / data['Russell'].shift(1)
btc_v = data['BTC'] / data['BTC'].shift(1)



# if goes up, 0 if decreases
djia = np.where(djia_v > 1, 1, 0) 
nasdaq = np.where(nasdaq_v > 1, 1, 0)
sp = np.where(sp_v > 1, 1, 0)
russell = np.where(russell_v > 1, 1, 0)
btc = np.where(btc_v > 1, 1, 0)

# random 1, 0
r = np.random.randint(2, size=len(djia))

# periodic
p = np.zeros(len(djia))
for i in range(len(p)):
    if i % 2 == 1:
        p[i] = 1




###############################################################################
# print bar code type plots
###############################################################################
def bar_plot(s):
    
    # https://matplotlib.org/examples/pylab_examples/barcode_demo.html
    axprops = dict(xticks=[], yticks=[])
    barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest')
    
    fig = plt.figure(figsize=(16, 4))
    x = s.copy()
    x.shape = 1, len(x)
    ax = fig.add_axes([0.3, 0.1, 0.6, 0.1], **axprops)
    ax.imshow(x, **barprops)
    
    plt.show()
    
 


###############################################################################
# random, structured, universal ?
###############################################################################


def law_of_large_numbers(s):
    
    total = len(s)
    ones = np.sum(s)
    zeros = total - ones
    
    
    print('Probability %.2f, %.2f ' % (ones/total, zeros/total ))
    




def central_limit_thm(s):
    
    m = np.mean(s)
    v = np.var(s)
    std = np.std(s)
    n = len(s)
    
    sk = np.sum((s - m)**3 / n) / (std**3)
    skw = (np.sqrt(n*(n-1)))/(n-2) * sk 
    
    kurt = np.sum((s - m)**4 / n) / (std**4)
    e_kurt = kurt - 3
    

    print('Mean %.2f ' % m)
    print('Var %.2f' % v)
    print('Std %.2f'% std)
    print('Skew %.2f, %.2f' % (sk, skw))
    print('Kurt %.2f, %.2f'% (kurt, e_kurt))

    




##############################################################################
# print info to screen
##############################################################################
print('----------------------------------------------------------------------')
print('Up days vs down days    July 16, 2010 - March 7, 2018')    
print('----------------------------------------------------------------------')   

print('DJIA')
law_of_large_numbers(djia)
central_limit_thm(djia)
bar_plot(djia)
print('------------------------------')   

print('NASDAQ')
law_of_large_numbers(nasdaq)
central_limit_thm(nasdaq)
bar_plot(nasdaq)
print('------------------------------')   

print('S&P')
law_of_large_numbers(sp)
central_limit_thm(sp)
bar_plot(sp)
print('------------------------------')   

print('Russell')
law_of_large_numbers(russell)
central_limit_thm(russell)
bar_plot(russell)
print('------------------------------')   

print('BTC')
law_of_large_numbers(btc)
central_limit_thm(btc)
bar_plot(btc)
print('------------------------------')   

print('Random')
law_of_large_numbers(r)
central_limit_thm(r)
bar_plot(r)
print('------------------------------')   

print('Periodic')
law_of_large_numbers(p)
central_limit_thm(p)
bar_plot(p)
print('------------------------------')   






















