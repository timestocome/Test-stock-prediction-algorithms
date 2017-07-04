# http://github.com/timestocome

# yaade
# Yet another anomaly detection experiment

import pandas as pd 
import numpy as np
from numpy import fft
import tensorflow as tf
import matplotlib.pyplot as plt



# read in datafile downloaded from Google.finance weird encoding
data = pd.read_csv('data/Fred_Nasdaq.csv', parse_dates=True, index_col=0, encoding='utf-8-sig')


# flip order so oldest is first
#data = data.iloc[::-1]

data['Open'] = pd.to_numeric(data['Open'], errors='coerce')
data = data.dropna()

# convert to log scale
data['Open'] = np.log(data['Open'])


# trim some dates
data = data.loc[data.index > '1990-01-01']



#####################################################################################
# find and remove trend line so time series is stationary 
x1 = data.iloc[0]['Open']
x2 = data.iloc[len(data)-1]['Open']

# dy/dx
dx = x2 - x1
dy = len(data)
dx_dy = dx/dy 

t = np.arange(0, len(data))
data['y'] = dx_dy * t + x1


data['StationaryOpen'] = data['Open'] - data['y']


# plot to see if it looks correct
plt.figure(figsize=(12, 12))
plt.plot(data['Open'], label='Data')
plt.plot(data['y'], label='Trend')
plt.plot(data['StationaryOpen'], label='Stationary')
plt.legend(loc='best')
plt.title('Nasdaq analysis')
plt.savefig("TimeSeriesAnomalies_LogStationary.png")
plt.show()

#######################################################################################
# look for cycles in data shorter than one year
from scipy.fftpack import rfft, irfft, fftfreq

time = t 
signal = data['StationaryOpen']

ps = np.abs(np.fft.fft(signal)) **2
frequencies = np.fft.fftfreq(len(data), 1)


import pylab as plt
plt.figure(figsize=(14,14))
plt.plot(t, ps)
plt.ylim(0, 1000)
plt.xlim(0, 253)        
plt.grid('on')
plt.xlabel('Trading days')
plt.ylabel('Cycle strength')
plt.title('Nasdaq FFT Power')
plt.savefig('TimeSeriesAnomlies_FFTCycles.png')
plt.show()

# get indexes of strongest frequencies
d_df = pd.DataFrame({'P': ps})

# ditch everything over 253 days long
p_df_short = d_df[1:253]

print("Strongest cycles:")
print("Days:\t Power:")
cycles = p_df_short.nlargest(10, 'P')
print(cycles)

###########################################################
# smooth out noise
window = 21     # 5 days in week, 21 in month, 63 in quarter, 253 in year

data['ma'] = data['StationaryOpen'].rolling(window=window, center=True).mean()

plt.figure(figsize=(14,14))
plt.plot(data['ma'], lw=10, alpha=0.3, label='Moving average')
plt.plot(data['StationaryOpen'], label='Data')
plt.legend(loc='best')
plt.title('Monthly moving average anomalies')
plt.savefig("TimeSeriesAnonalies_Monthly.png")
plt.show()



#######################################################################################
# look for cycles again after smoothing data 

signal = data['ma']
signal = signal.dropna()
t = np.arange(0, len(data))

ps = np.abs(np.fft.fft(signal)) **2
frequencies = np.fft.fftfreq(len(data), 1)



plt.figure(figsize=(14,14))
plt.plot(ps)
plt.ylim(0, 1000)
plt.xlim(0, 253)        
plt.grid('on')
plt.xlabel('Trading days')
plt.ylabel('Cycle strength')
plt.title('Nasdaq FFT Power on smoothed data')
plt.savefig('TimeSeriesAnomlies_SmoothedFFTCycles.png')
plt.show()

# get indexes of strongest frequencies
d_df = pd.DataFrame({'P': ps})

# ditch everything over 253 days long
p_df_short = d_df[1:253]

print("Strongest cycles on smoothed data:")
print("Days:\t Power:")
cycles = p_df_short.nlargest(10, 'P')
print(cycles)