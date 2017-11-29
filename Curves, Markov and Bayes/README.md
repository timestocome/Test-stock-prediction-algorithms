

# Stock Market Curves and Markov Chains

### Load and Match Dates 
Takes data downloaded from finance.yahoo.com and grabs daily Open and Volume and matches dates across DJI, NASDAQ and S&P index funds and stores it in StockDataWithVolume.csv

### Level Data
Takes StockDataWithVolume.csv computes volatility and levels time series stores data in LeveledLogStockData.csv

### Stock Curves
Takes StockDataWithVolume.csv and computes histograms of daily gains, losses and plots them, stores it in histogram.png




## ToDo:
### Markov Chains
Take gains and losses and look for various length patterns


### Bayesian Analysis
Take gains and losses and recursively look for promising predictions
