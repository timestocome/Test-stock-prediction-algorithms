
# http://github.com/timestocome

# Level data so series is stationary in time
# take log of data
# save it to use in deconstructing signal to find anomolies 
# 
# https://blog.statsbot.co/time-series-anomaly-detection-algorithms-1cef5519aef2


# nice plot of anomalies, but no new information here

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


# pandas display options
pd.options.display.max_rows = 100
pd.options.display.max_columns = 25
pd.options.display.width = 1000



######################################################################
# plot dataframe
########################################################################
def plot_dataframe(d, t):


    plt.figure(figsize=(18,18))
    plt.plot(d['NASDAQ'], label='NASDAQ')
    plt.plot(d['S&P'], label='S&P')
    plt.plot(d['Russell 2000'], label='Russell')
    plt.plot(d['DJIA'], label='DJIA')
    plt.plot(d['Gold'], label='Gold')
    plt.plot(d['1yr Treasury'], label='1yr T')
    plt.plot(d['10yr Bond'], label='10 yr bond')
    plt.plot(d['VIX'], label='VIX')
    plt.title(t)
    plt.legend(loc='best')
    plt.show()


######################################################################
# data
########################################################################
# data is already converted to a log scale and leveled ( using lr ) 
# see https://github.com/timestocome/StockMarketData for the code and data csv file

scaled_data = pd.read_csv('LeveledLogStockData.csv', index_col='Date', parse_dates=True)
scaled_data = scaled_data.fillna(method='ffill')
scaled_data.columns = ['Nasdaq', 'S&P', 'Russell 2000', 'DJIA', 'Gold', '1yr T', '1yr Bond']


# Cleaned original data
# see https://github.com/timestocome/StockMarketData for the code and data csv file
data = pd.read_csv('StockDataWithVolume.csv', index_col='Date', parse_dates=True)


##############################################################################
# Calculate largest gains/losses dates on NASDAQ

data['log nasdaq'] = np.log(data['NASDAQ'])
data['nasdaq volatility'] = data['log nasdaq'] - data['log nasdaq'].shift(1)

gains = data['nasdaq volatility'].nlargest(50)
losses = data['nasdaq volatility'].nsmallest(50)



##############################################################################

# calculate daily change on leveled, log data
dailyVolatility = scaled_data['Nasdaq'] - scaled_data['Nasdaq'].shift(1)
dailyVolatility = dailyVolatility.dropna()


dailyVolatility = dailyVolatility * dailyVolatility
largestVolatilityDays = dailyVolatility.nlargest(100)


##############################################################################
# check to see if other indexes might be leading indicators
# meh - none of these are leading indicators of NASDAQ anomalies

'''
# not a leading indicator 
data['log vix'] = np.log(data['VIX'])
data['vix volatility'] = data['log vix'] - data['log vix'].shift(1)
vix_gains = data['vix volatility'].nlargest(50)
vix_losses = data['vix volatility'].nsmallest(50)


data['log gold'] = np.log(data['Gold'])
data['gold volatility'] = data['log gold'] - data['log gold'].shift(1)
gold_gains = data['gold volatility'].nlargest(50)
gold_losses = data['gold volatility'].nsmallest(50)


data['log T'] = np.log(data['1yr Treasury'])
data['T volatility'] = data['log T'] - data['log T'].shift(1)
T_gains = data['T volatility'].nlargest(50)
T_losses = data['T volatility'].nsmallest(50)


data['log B'] = np.log(data['10yr Bond'])
data['B volatility'] = data['log B'] - data['log B'].shift(1)
B_gains = data['B volatility'].nlargest(50)
B_losses = data['B volatility'].nsmallest(50)


data['log Russell'] = np.log(data['Russell 2000'])
data['Russell volatility'] = data['log Russell'] - data['log Russell'].shift(1)
russell_gains = data['Russell volatility'].nlargest(50)
russell_losses = data['Russell volatility'].nsmallest(50)
'''

############################################################################

scaled_data['highest gains'] = gains
scaled_data['worst losses'] = losses
scaled_data['highest volatility'] = largestVolatilityDays

inflections = scaled_data[['highest gains', 'worst losses', 'highest volatility']]
inflections = inflections.fillna(0)

# plot other information see if it leads NASDAQ big days
#inflections['Russell_up'] = russell_gains
#inflections['Russell_down'] = russell_losses

plt.figure(figsize=(18, 16))
plt.title("Nasdaq anomalies")
plt.plot(inflections['highest gains'], label='Gains')
plt.plot(inflections['worst losses'], label='Losses')
plt.plot(inflections['highest volatility'], label='Volatility', lw=2)
#plt.plot(inflections['Russell_up'], label='Russell up', lw=5)
#plt.plot(inflections['Russell_down'], label='Russell down', lw=5)
plt.legend(loc='best')

plt.savefig('nasdaq_anomalies.png')
plt.show()