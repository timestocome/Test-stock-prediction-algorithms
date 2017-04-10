# http://github.com/timestocome




# http://pandas.pydata.org 
import pandas as pd

pd.options.display.max_rows = 100
pd.options.display.max_columns = 25
pd.options.display.width = 1000

import numpy as np


# number of peak gains/losses to pull
n_gain_loss = 5000


##########################################################################
# data is from  https://www.measuringworth.com/datasets/DJA/index.php
##########################################################################
# read in DJA
dja = pd.read_csv('DJA.csv')        # 31747 days of data 
n_samples = len(dja)

# get more details on the date
dja['Year'] = pd.DatetimeIndex(dja['Date']).year 
dja['Month'] = pd.DatetimeIndex(dja['Date']).month
dja['Day'] = pd.DatetimeIndex(dja['Date']).day
dja['DayOfWeek'] = pd.DatetimeIndex(dja['Date']).dayofweek   # Monday is day 0
dja['DayOfYear'] = pd.DatetimeIndex(dja['Date']).dayofyear   # Day of year
dja['Quarter'] = pd.DatetimeIndex(dja['Date']).quarter       # quarter 

# daily change in dja (derivative)
dja['dx'] = dja['DJIA'] - dja['DJIA'].shift(1)      # points
dja['percent_dx'] = dja['dx'] / dja['DJIA']         # percent

# 2nd derviative
dja['d2x'] = dja['dx'] - dja['dx'].shift(1) 
dja['percent_d2x'] = dja['d2x'] / dja['DJIA']



# scale djia
min_djia = dja['DJIA'].min()
max_djia = dja['DJIA'].max()
dja['Scaled_DJIA'] = (dja['DJIA'] - min_djia ) / ( max_djia - min_djia )



# one hot date vectors (month, day, dayofweek, quarter, year)
# quarter
dja['Q1'] = np.where(dja['Quarter'] == 1, 1, 0)
dja['Q2'] = np.where(dja['Quarter'] == 2, 1, 0)
dja['Q3'] = np.where(dja['Quarter'] == 3, 1, 0)
dja['Q4'] = np.where(dja['Quarter'] == 4, 1, 0)

# day of week
dja['DoW1'] = np.where(dja['DayOfWeek'] == 1, 1, 0)
dja['DoW2'] = np.where(dja['DayOfWeek'] == 2, 1, 0)
dja['DoW3'] = np.where(dja['DayOfWeek'] == 3, 1, 0)
dja['DoW4'] = np.where(dja['DayOfWeek'] == 4, 1, 0)
dja['DoW5'] = np.where(dja['DayOfWeek'] == 5, 1, 0)


# month
dja['M1'] = np.where(dja['Month'] == 1, 1, 0)
dja['M2'] = np.where(dja['Month'] == 2, 1, 0)
dja['M3'] = np.where(dja['Month'] == 3, 1, 0)
dja['M4'] = np.where(dja['Month'] == 4, 1, 0)
dja['M5'] = np.where(dja['Month'] == 5, 1, 0)
dja['M6'] = np.where(dja['Month'] == 6, 1, 0)
dja['M7'] = np.where(dja['Month'] == 7, 1, 0)
dja['M8'] = np.where(dja['Month'] == 8, 1, 0)
dja['M9'] = np.where(dja['Month'] == 9, 1, 0)
dja['M10'] = np.where(dja['Month'] == 10, 1, 0)
dja['M11'] = np.where(dja['Month'] == 11, 1, 0)
dja['M12'] = np.where(dja['Month'] == 12, 1, 0)



min_year = dja['Year'].min()
dja['Adjusted_Year'] = dja['Year'] - min_year
max_scaled_year = dja['Adjusted_Year'].max()
print(max_scaled_year)

# create new year column for each year
years = []

for i in range(max_scaled_year):
    y = 'Y' + str(i)            # create name for column
    years.append(y)             # add to column list
    dja[y] = 0                  # init new column to zero


# convert year
def one_hot_year(yr):
    adj_yr = yr - min_year 
    y = 'Y' + str(adj_yr)
    dja[y] = 1

dja['Year'].apply(one_hot_year) 



# drop first few rows where dx/dy == nan
dja = dja.dropna()


# sanity check
features = ['percent_dx', 'percent_d2x', 'Scaled_DJIA', 'Q1', 'Q2', 'Q3', 'Q4', 'DoW1', 'DoW2', 'DoW3', 'DoW4', 'DoW5', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12']
features = features + years
print(features)
dja = dja[features]
print(dja)


# write to disk
dja.to_csv("cleaned_dja.csv")

