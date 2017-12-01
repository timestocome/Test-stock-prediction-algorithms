

# Stock Market Curves and Markov Chains

### Load and Match Dates 
Takes data downloaded from finance.yahoo.com and grabs daily Open and Volume and matches dates across DJI, NASDAQ and S&P index funds and stores it in StockDataWithVolume.csv

### Level Data
Takes StockDataWithVolume.csv computes volatility and levels time series stores data in LeveledLogStockData.csv

### Stock Curves
Takes StockDataWithVolume.csv and computes histograms of daily gains, losses and plots them, stores it in histogram.png




## ToDo:
### Markov Chains
The state machine builder code is written, hope to get the rest of the Markov done next week (1st week Dec 17)
Take gains and losses and look for various length patterns


### Bayesian Analysis
The state machine builder code is done, planning on tackling this after I finish the Markov Chains
Take gains and losses and recursively look for promising predictions
