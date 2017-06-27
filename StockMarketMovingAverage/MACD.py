# https://github.com/timestocome
# http://www.mrao.cam.ac.uk/~mph/Technical_Analysis.pdf


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


# moving average convergence/divergence by Appel in 1970s
# used to spot changes in strength, direction, momentum, and duration of price
# acts as a moving filter of velocity

# lagging indicator
# uses actual prices, not logs or percent changes
# 

# macd = ma_12_day - ma_26_day  # blue line
# signal = ma_9_day               # red line
# histogram = macd - signal 


# macd - signal = 0  # buy when crosses up, sell when crosses down
# ma_12_day - ma_26_day = 0
# sign(final_price - initial_price) != sign(macd_last - macd_init)



print("Test buying / selling on moving average with $10,000 seed money")
print("Test fixed dollar purchases between 1980-2016")



# select years
start_year = 1980           # data is 1900-2017 pick year to start training 
end_year = 2016

# bot stuff
seed_money = 10000.         # starting cash for bots
commission = 7.             # flat commission per trade

# moving average window sizes commonly used for this algorithm
signal_window = 9     
macd_window = 12


###################################################################################
# trading bot
##################################################################################

class bot():

    def __init__(self):

        self.shares_on_hand = 0.
        self.cash_on_hand = seed_money
        self.commission = commission


  
    def buy(self, current_price):

        if self.cash_on_hand > 0.:

            self.shares_on_hand = (self.cash_on_hand - self.commission) / current_price
            self.cash_on_hand = 0.  



    def sell(self, current_price):

        if self.shares_on_hand > 0.:

            self.cash_on_hand = (self.shares_on_hand * current_price) - commission
            self.shares_on_hand = 0.



    def cash_out(self, current_price):
        
        if self.shares_on_hand > 0.:
            cash_from_sale = (self.shares_on_hand * current_price) - self.commission

            self.shares_on_hand = 0
            self.cash_on_hand += cash_from_sale


##########################################################################
# data is from  https://www.measuringworth.com/datasets/DJA/index.php
# and http://finance.yahoo.com 
##########################################################################

# read in NASDAQ
stock_data = pd.read_csv('data/nasdaq.csv', parse_dates=True, index_col=0)
stock_data = stock_data[['Close']]
stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')


#######################################################################
# algorithm 
######################################################################

# for plotting
profit = []         # cash on hand at end
fees = []           # trading fees ( commissions )
max_net = 0.


# calculate moving avg, crossovers and over/under
stock_data['signal'] = stock_data['Close'].rolling(window=signal_window).mean()
stock_data['macd'] = stock_data['Close'].rolling(window=macd_window).mean() 

stock_data['buy'] = np.where(stock_data['macd'] > stock_data['signal'], 1, 0)
stock_data['sell'] = np.where(stock_data['signal'] > stock_data['macd'].shift(1), 1, 0)

stock_data = stock_data.dropna()



trader = bot()
last_price = stock_data.iloc[0]['Close']


for ix, row in stock_data.iterrows():

    if row['buy'] == 1:         # buy
        if trader.cash_on_hand > 0:
            trader.buy(row['Close'])
            print("Buy: %s     Shares: %lf" %(row.name, trader.shares_on_hand))

    
    elif row['sell']:           # sell
        if trader.shares_on_hand > 0:
            trader.sell(row['Close'])
            print("Sell: %s     $%.2lf" %(row.name, trader.cash_on_hand))


    last_price = row['Close']  # use to calculate buy and hold cash out amount


# cash out moving average bot
trader.cash_out(stock_data.iloc[-1]['Close'])


print("Final cash on hand: $%.2lf" % trader.cash_on_hand)
