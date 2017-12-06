

# Stock Market Curves and Markov Chains

### Load and Match Dates 
Takes data downloaded from finance.yahoo.com and grabs daily Open and Volume and matches dates across DJI, NASDAQ and S&P index funds and stores it in StockDataWithVolume.csv

### Level Data
Takes StockDataWithVolume.csv computes volatility and levels time series stores data in LeveledLogStockData.csv

### Stock Curves
Takes StockDataWithVolume.csv and computes histograms of daily gains, losses and plots them, stores it in histogram.png

### Markov Chains
Build a simple 1 dimensional Markov Chain using daily volatility and predict volatility for next several days


### ToDo
### Bayesian Analysis
Take gains and losses and recursively look for promising predictions
