# http://github.com/timestocome

# see what, if anything can be predicted with bayes rule

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


######################################################################
# data
########################################################################
# read in datafile created in LoadAndMatchDates.py
data = pd.read_csv('StockDataWithVolume.csv', index_col='Date', parse_dates=True)
features = [data.columns.values]

# switch to log data
log_data = np.log(data)

# predict tomorrow's gain/loss given today's information
log_data['target'] = log_data['NASDAQ'].shift(-1)

features = ['NASDAQ', 'VIX', 'US GDP', 'Gold', '1yr Treasury', '10yr Bond', 'UnEmploy', 'NASDAQ_Volume']


# log returns ( change in value ) for all features
log_change = log_data - log_data.shift(1)


# clean up 
log_data.dropna()
log_change.dropna()

n_samples = len(log_data)


# increase/decrease each day over previous day for all columns
p_up = (log_change > 0.).astype(int)
p_down = (log_change < 0.).astype(int)


# probabilities
p_nasdaq_up_tomorrow = p_up.sum()['target'] / n_samples 
p_nasdaq_down_tomorrow = p_down.sum()['target'] / n_samples



##############################################################################################
for f in features:

    # probabilities of indicators and tomorrow's nasdaq change 
    # try gold, count number of times both (1 + 1 = 2) appears in column
    nasdaq_up_indicator_up = (p_up['target'] + p_up[f]).value_counts()[2]
    nasdaq_up_indicator_down = (p_up['target'] + p_down[f]).value_counts()[2]
    nasdaq_d_indicator_up = (p_down['target'] + p_up[f]).value_counts()[2]
    nasdaq_d_indicator_down = (p_down['target'] + p_down[f]).value_counts()[2]

    nasdaq_d_sum = nasdaq_d_indicator_down + nasdaq_d_indicator_up
    nasdaq_up_sum = nasdaq_up_indicator_down + nasdaq_up_indicator_up
    indicator_d_sum = nasdaq_d_indicator_down + nasdaq_up_indicator_down
    indicator_up_sum = nasdaq_d_indicator_up + nasdaq_up_indicator_up



    # bayes probabilty
    p_stock_down_given_indicator_up = (nasdaq_d_sum / n_samples) * (nasdaq_d_indicator_up/nasdaq_d_sum) / (indicator_up_sum/n_samples)
    p_stock_up_given_indicator_up = (nasdaq_up_sum/n_samples) * (nasdaq_up_indicator_up/nasdaq_up_sum) / (indicator_up_sum/n_samples)
    p_stock_down_given_indicator_down = (nasdaq_d_sum/n_samples) * (nasdaq_d_indicator_down/nasdaq_d_sum) / (indicator_d_sum/n_samples)
    p_stock_up_given_indicator_down = (nasdaq_up_sum/n_samples) * (nasdaq_up_indicator_down/nasdaq_up_sum) / (indicator_d_sum/n_samples)

    print("Probability Nasdaq up tomorrow if %s up today: %f" % (f, p_stock_up_given_indicator_up))
    print("Probability Nasdaq down tomorrorw if %s down today: %f" % (f, p_stock_down_given_indicator_up))
    print("Probability Nasdaq up tomorrow if %s down today: %f" % (f, p_stock_up_given_indicator_down))
    print("Probability Nasdaq down tomorrow if %s down today: %f" %  (f, p_stock_down_given_indicator_down))
    print("--------------------------------------------------------------------------------------------")



##############################################################################################
