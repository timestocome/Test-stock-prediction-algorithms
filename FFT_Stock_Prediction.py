# http://github.com/timestocome

# Try using Fourier Series to predict market indexes
# 3 harmonics seems to do a good job of predicting next quarter


import pandas as pd 
import numpy as np
from numpy import fft
import tensorflow as tf
import matplotlib.pyplot as plt



# read in datafile created in LoadAndMatchDates.py
data = pd.read_csv('StockDataWithVolume.csv', index_col='Date', parse_dates=True)
features = ['DJIA', 'S&P', 'Russell 2000', 'NASDAQ', 'VIX', 'Gold']



# frequency/FFT
# https://gist.github.com/tartakynov/83f3cd8f44208a1856ce
def fourierEx(x, n_predict, harmonics):

    n = len(x)                  # number of input samples
    n_harmonics = harmonics            # f, 2*f, 3*f, .... n_harmonics  ( 1,2, )
    t = np.arange(0, n)         # place to store data
    p = np.polyfit(t, x, 1)     # find trend
    x_no_trend = x - p[0] * t 
    x_frequency_domains = fft.fft(x_no_trend)
    f = np.fft.fftfreq(n)       # frequencies
    indexes = list(range(n))
    indexes.sort(key=lambda i: np.absolute(f[i]))

    t = np.arange(0, n + n_predict)
    restored_signal = np.zeros(t.size)
    for i in indexes[:1 + n_harmonics * 2]:
        amplitude = np.absolute(x_frequency_domains[i] / n)
        phase = np.angle(x_frequency_domains[i])
        restored_signal += amplitude * np.cos(2 * np.pi * f[i] * t + phase)
    
    return restored_signal + p[0] * t 



def plot_fft_djia():

    # pull out hold out data 
    x = data.loc[data.index < '06-30-2016']     # pull out 6 months data to check prediction
    x = x['DJIA']
    z = data['DJIA']
    
    # match this to hold out data count to properly line up plot lines
    n_predict = 126         # number of data points to predict (5 week, 21 month, 63 quarter, 126 half year)

    fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(20,16))
    axs = axs.ravel()
    plt.setp(axs, xticks=np.arange(1, 7000, step=1300), xticklabels=np.arange(1990, 2016, step=5))


    i = 0
    
    
    
    for h in range(1,13):

        extrapolation = fourierEx(x, n_predict, h)

        axs[i].set_title("DJIA FFT projection, harmonics %d" %(h))
        axs[i].plot(np.arange(len(extrapolation)), extrapolation, 'r', label='extrapolation', linewidth=2)
        axs[i].plot(np.arange(len(z)), z, 'b', label='DJIA', linewidth=2)
        axs[i].plot(np.arange(len(x)), x, 'darkgray', label='training data', linewidth=1)

        i += 1


    
    plt.legend(loc=0)
    plt.savefig("FFT_DJIA.png")
    plt.show()




plot_fft_djia()
