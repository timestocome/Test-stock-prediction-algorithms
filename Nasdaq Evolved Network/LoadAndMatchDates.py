# http://github.com/timestocome
#
# load in data from finance.yahoo.com
# match dates
# 1985-2107 Nasdaq, Dji, S&P
# saves opening price, volume for each day market is open



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 




# pandas display options
pd.options.display.max_rows = 100
pd.options.display.max_columns = 25
pd.options.display.width = 1000





# read in file
def read_data(file_name):

    stock = pd.read_csv(file_name, parse_dates=True, index_col=0)     
    
    # ditch samples with NAN values
    stock = stock.dropna(axis=0)

    # flip order from newest to oldest to oldest to newest
    #stock = stock.iloc[::-1]

    # trim data
    stock = stock[['Open', 'Volume']]

    # convert object to floats
    stock['Open'] = pd.to_numeric(stock['Open'], errors='coerce')

    # all stock is needed to walk back dates for testing hold out data
    return stock


#############################################################################################
# load and combine stock indexes, matching the dates 


def load_and_combine_data():

    dow_jones = read_data('DJI.csv')
    s_p = read_data('GSPC.csv')
    nasdaq = read_data('IXIC.csv')
    btc = read_data('BTC-USD.csv')
    russell = read_data('RUT.csv')

  
    # rename columns before joining so we know which is which
    dow_jones.columns = ['DJIA', 'DJIA_Volume']
    s_p.columns = ['S&P', 'S&P_Volume']
    nasdaq.columns = ['NASDAQ', 'NASDAQ_Volume']
    btc.columns = ['BTC', 'BTC_Volume']
    russell.columns = ['Russell', 'Russell_Volume']

    
    
    # combine by matching date index, missing dates will get NaN
    indexes = dow_jones.join(s_p)
    indexes = indexes.join(btc)
    indexes = indexes.join(russell)
    indexes = indexes.join(nasdaq)
    
    indexes = indexes.dropna()

    '''
    # compare indexes
    test = indexes[['DJIA', 'S&P', 'NASDAQ', 'BTC', 'Russell']]
    (test / test.iloc[0] * 100).plot(figsize=(20,15))
    plt.title("Standarized Indexes 1985-2017")
    plt.savefig("StandardizedData.png")
    plt.show()
    '''

    # save file
    indexes.to_csv("StockDataWithVolume.csv")

    return (indexes)





df = load_and_combine_data()
