# http://github.com/timestocome
#
# load in data from finance.yahoo.com
# match dates
# 1985-2107 Nasdaq, Dji, S&P
# saves opening price, volume for each day market is open



import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import BayesianRidge, LinearRegression
import matplotlib.pyplot as plt 




# pandas display options
pd.options.display.max_rows = 100
pd.options.display.max_columns = 25
pd.options.display.width = 1000





# read in file
def read_data(file_name):

    stock = pd.read_csv(file_name, parse_dates=True, index_col=0)     
    n_samples = len(stock)
    
    # ditch samples with NAN values
    stock = stock.dropna(axis=0)

    # flip order from newest to oldest to oldest to newest
    #stock = stock.iloc[::-1]

    # trim data
    stock = stock[['Open']]

    # convert object to floats
    stock['Open'] = pd.to_numeric(stock['Open'], errors='coerce')

    # all stock is needed to walk back dates for testing hold out data
    return stock


#############################################################################################
# load and combine stock indexes, matching the dates 


def load_and_combine_data():

    dow_jones = read_data('DJI.csv')
    s_p = read_data('SP500.csv')
    nasdaq = read_data('NASDAQ.csv')

  
    # rename columns before joining so we know which is which
    dow_jones.columns = ['DJIA']
    s_p.columns = ['S&P']
    nasdaq.columns = ["NASDAQ"]

    
    # combine by matching date index, missing dates will get NaN
    indexes = dow_jones.join(s_p)
    indexes = indexes.join(nasdaq)


    # compare indexes
    (indexes / indexes.ix[0] * 100).plot(figsize=(20,15))
    plt.title("Standarized Indexes 1985-2017")
    plt.savefig("StandardizedData.png")
    plt.show()
    

    # save file
    indexes.to_csv("StockData.csv")

    return (indexes)




def add_volume(df):

    # read in data files containing Volume data
    dj = pd.read_csv('DJI.csv', parse_dates=True, index_col=0)  
    sp = pd.read_csv('SP500.csv', parse_dates=True, index_col=0)
    nasdaq = pd.read_csv('NASDAQ.csv', parse_dates=True, index_col=0)

    # pull out volume columns and add to dataframe
    df['DJIA_Volume'] = dj['Volume']
    df['S&P_Volume'] = sp_volume = sp['Volume']
    df['NASDAQ_Volume'] = nasdaq['Volume']

    print(df)

    # save file
    df.to_csv('StockDataWithVolume.csv')


df = load_and_combine_data()
add_volume(df)
