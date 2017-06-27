# https://github.com/timestocome
# http://www.mrao.cam.ac.uk/~mph/Technical_Analysis.pdf


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


# Mass Index by Donald Dorsey to predict trend reversals



##########################################################################
# data is from  https://www.measuringworth.com/datasets/DJA/index.php
# and http://finance.yahoo.com 
##########################################################################

# read in NASDAQ
data = pd.read_csv('data/nasdaq.csv', parse_dates=True, index_col=0)
data = data[['High', 'Low', 'Open']]
data['Open'] = pd.to_numeric(data['Open'], errors='coerce')
data['High'] = pd.to_numeric(data['High'], errors='coerce')
data['Low'] = pd.to_numeric(data['Low'], errors='coerce')


# high - low
data['Diff'] = data['High'] - data['Low']

# 9 day EMA of high - low
data['EMA9'] = data['Diff'].ewm(span=9).mean()

# daily mass
data['Mass'] = data['EMA9'] / (data['EMA9']).ewm(span=9).mean()

# 25 day sum
data['Mass25'] = data['Mass'].rolling(window=25).sum()

# clean up
data = data.dropna()


# use log for plotting
data['LogNasdaq'] = np.log(data['Open'])

####################################################################
plt.figure(figsize=(16,16))
plt.title("Mass Index vs Nasdaq")
plt.plot(data['Open'], label='Nasdaq')
plt.plot(data['Mass25'] * 100, label="MassIndex")
plt.legend(loc='best')
plt.grid('on')
plt.savefig('MassIndex.png')
plt.show()