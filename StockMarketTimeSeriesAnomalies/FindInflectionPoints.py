

# http://github.com/timestocome

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import signal

# use data leveled, log'd and otherwise smoothed in 
# https://github.com/timestocome/StockMarketData
# to do some analysis

# http://www.mdpi.com/1999-4893/5/4/588
# after I started working through the algorithm
# it became clear it's not so different than convolution
# and convolution might be easier and faster so shifted 
# to using the built in scipy.signal library
# The signal still needs to be stationary (rotated to x axis) in time
# and for stocks because of inflation you'll need a log index or the 
# older ones will be too small to catch 
#
# to find the bottoms of the Nasdad flip signal around the x axis and 
# repeat


# import data that we've rotated to x axis to make stationary in time (see section 1 of above paper)
# and scaled by taking the log
data = pd.read_csv('LeveledLogStockData.csv', index_col=0, parse_dates=True)

features = ['Nasdaq', 'S&P', 'Russell', 'DJIA', 'Gold', '1yr T', '10yr Bond']
data.columns = ['Nasdaq', 'S&P', 'Russell', 'DJIA', 'Gold', '1yr T', '10yr Bond']



for f in features:
    inverted_name = 'Flipped_' + f
    peaks_name = 'Peaks_' + f
    floors_name = 'Floors_' + f 

    inverted_signal = data[f] * -1.

    peaks_ix = signal.find_peaks_cwt(data[f], np.arange(1, 253))
    peaks = np.zeros(len(data))
    for i in peaks_ix: peaks[i] = 1
    data[peaks_name] = peaks 

    floor_ix = signal.find_peaks_cwt(data[f], np.arange(1, 253))
    floors = np.zeros(len(data))
    for i in floor_ix: floors[i] = 1 
    data[floors_name] = floors 



inflection_dates = ['Peaks_Nasdaq', 'Floors_Nasdaq','Peaks_S&P', 'Floors_S&P', 'Peaks_Russell', 'Floors_Russell', 'Peaks_DJIA', 
        'Floors_DJIA', 'Peaks_Gold', 'Floors_Gold', 'Peaks_1yr T', 'Floors_1yr T', 'Peaks_10yr Bond', 'Floors_10yr Bond']


data[inflection_dates].to_csv("inflectionDates.csv") 



