

# Stock Market Curves and Markov Chains

### Load and Match Dates 
Takes data downloaded from finance.yahoo.com and grabs daily Open and Volume and matches dates across DJI, NASDAQ and S&P index funds and stores it in StockDataWithVolume.csv

### Level Data
Takes StockDataWithVolume.csv computes volatility and levels time series stores data in LeveledLogStockData.csv

### Stock Curves
Takes StockDataWithVolume.csv and computes histograms of daily gains, losses and plots them, stores it in histogram.png

### Markov Chains
More info:

https://hackernoon.com/from-what-is-a-markov-model-to-here-is-how-markov-models-work-1ac5f4629b71

See also Discrete-time Markov Chains 

https://en.wikipedia.org/wiki/Markov_chain

A simple 1 dimensional Markov Chain using daily volatility and predict volatility for next several days:

PredictMarketWithMarkovChain.py


### Bayesian Analysis
A very simple prediction model using 2 days movements

* Both the Bayes and Markov mostly predict the same most likely movements for the next day given today's state. Remember past performance and future performance may have nothing to do with each other.


### Tracy Widom 
I thought I'd find some Tracy-Widom curves but BitCoin turned out Platykurtic instead

Tracy Widom

https://www.quantamagazine.org/beyond-the-bell-curve-a-new-universal-law-20141015/

https://pdfs.semanticscholar.org/68bd/1b58980d4c5831b46725256aaee600be1734.pdf

Platykurtic

... still looking for good info
