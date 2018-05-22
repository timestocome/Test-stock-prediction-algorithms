#  http://github.com/timestocome


# read in bitcoin daily price from https://coinmetrics.io/data-downloads/
# save date, price as pandas series


import numpy as np
import pandas as pd
from datetime import datetime




class Data:
    
    def __init__(self, filename):
        
        data = pd.read_csv(filename, parse_dates=True)
        data.index.name = 'Date'
        data.columns = ['Tranactions', 'Count', 'Cap', 'Price', 'Volume', 'New Coins', 'fees', 'addresses', 'x']
        data['log price'] = np.log(data['Price'])        

        
        data['dx'] = data['log price'] - data['log price'].shift(1)
                
        
        data = data[['Price', 'dx']]        
        data = data.dropna()
        
        self.data = data   
        self.n_rows = data.shape[0]

        
        
    def price_at_index(self, index):
        
        price = self.data.iloc[index]
        return price
        
        
    def price_at_date(self, month, day, year):
        
        date = datetime(year, month, day)
        price = self.data.loc[date]
        
        return price
        
        
        
        
'''       
###############################################################################
# run tests
if __name__ == '__main__':
        
    daily = Data('btc.csv')


    price = daily.price_at_index(5)
    print(price)

    month = 5
    day = 10
    year = 2013
    price = daily.price_at_date(month, day, year)
    print(price)
'''



#signal = Data('btc.csv')
#print(signal.data)



