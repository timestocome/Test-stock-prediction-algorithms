# http://github.com/timestocome




import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt 



# pandas display options
pd.options.display.max_rows = 100
pd.options.display.max_columns = 25
pd.options.display.width = 1000




# files
dow = 'data/DOW30.csv'
nasdaq = 'data/nasdaq.csv'
russell = 'data/Russell2000.csv'
sp = 'data/S&P.csv'

gdp = 'data/GDP_US.csv'
gold = 'data/GOLD.csv'
vix = 'data/VIX.csv'



# read in file
def read_data(file_name):

    stock = pd.read_csv(file_name, parse_dates=True, index_col=0)        # 31747 days of data 
    n_samples = len(stock)

    return stock




# read in csv files
dj_stock = read_data(dow)
n_stock = read_data(nasdaq)
r_stock = read_data(russell)
sp_stock = read_data(sp)

#print(dj_stock)



# features to keep
features = ['Open', 'Volume']



# combine data series into one table
data = pd.concat([dj_stock[features], n_stock[features], r_stock[features], sp_stock[features]], join='outer', axis=1)
#print(data)



# rename columns
data.columns = ['DJ Open', 'DJ Volume', 'Nasdaq Open', 'Nasdaq Volume', 'Russell Open', 'Russell Volume', 'S&P Open', 'S&P Volume']
#print(data)



# add volatility
data['DJ Log Ret'] = np.log(data['DJ Open']/data['DJ Open'].shift(1))
data['N Log Ret'] = np.log(data['Nasdaq Open']/data['Nasdaq Open'].shift(1))
data['R Log Ret'] = np.log(data['Russell Open']/data['Russell Open'].shift(1))
data['SP Log Ret'] = np.log(data['S&P Open']/data['S&P Open'].shift(1))

w = 63 # ~ 1 year
data['DJ Volatility'] = data['DJ Log Ret'].rolling(window=w, center=True).std() 
data['N Volatility'] = data['N Log Ret'].rolling(window=w, center=True).std() * np.sqrt(252)
data['R Volatility'] = data['R Log Ret'].rolling(window=w, center=True).std() * np.sqrt(252)
data['SP Volatility'] = data['SP Log Ret'].rolling(window=w, center=True).std() * np.sqrt(252)



# drop all rows with NAN
data = data.dropna(axis=0)      # 0 for rows, 1 is for columns
#print(data)



# plot data
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 18))

fig.suptitle("Index, Volatility, Volume")
axes[0], axes[1], axes[2], axes[3] = axes.flatten()

axes[0].plot(data['DJ Open'].rolling(window=w, center=True).mean(), label="DJ")
axes[0].plot(data['DJ Volatility'].rolling(window=w, center=True).mean() * 100000, label="Volatility")
axes[0].plot(data['DJ Volume'].rolling(window=w, center=True).mean()/100000, label="Volume")
axes[0].set_title("Dow Jones")
axes[0].legend(loc='upper left')



axes[1].plot(data['Nasdaq Open'].rolling(window=w, center=True).mean(), label="Nasdaq")
axes[1].plot(data['N Volatility'].rolling(window=w, center=True).mean() * 1000, label="Volatility")
axes[1].plot(data['Nasdaq Volume'].rolling(window=w, center=True).mean()/1000000, label="Volume")
axes[1].set_title("NASDAQ")
axes[1].legend(loc='upper left')




axes[2].plot(data['S&P Open'].rolling(window=w, center=True).mean(), label="S&P 2000")
axes[2].plot(data['SP Volatility'].rolling(window=w, center=True).mean() * 1000, label="Volatility")
axes[2].plot(data['S&P Volume'].rolling(window=w, center=True).mean()/10000000, label="Volume")
axes[2].set_title("S&P")
axes[2].legend(loc='upper left')



axes[3].plot(data['Russell Open'].rolling(window=w, center=True).mean(), label="Russell 2000")
axes[3].plot(data['R Volatility'].rolling(window=w, center=True).mean() * 1000, label="Volatility")
axes[3].plot(data['Russell Volume'].rolling(window=w, center=True).mean()/100000, label="Volume")
axes[3].set_title("Russell")
axes[3].legend(loc='upper left')



#data[['DJ Open', 'Nasdaq Open', 'Russell Open', 'S&P Open']].plot()
plt.savefig("Volatility.png")
plt.show()

