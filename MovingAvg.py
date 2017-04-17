# http://github.com/timestocome



# test buy when MA crosses under share price, sell when share price over MA
# 
# this is the first test draft, all I had time to do today




# todo:
# try all MA from one day to one year ~250 days
# try RL bots and see how ML does
# try GA and see how bots do
#
# plot best worst strategies 
# print monthly, yearly gains and losses for best and worst strategies




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 




# select years
start_year = 1980           # data is 1900-2017 pick year to start training 
end_year = 2016


seed_money = 10000.         # starting cash for bots
commission = 7.             # flat commission per trade
fixed_rate = 100.           # fixed dollar bot 
fixed_share = 1.            # fixed share bot

###################################################################################
# trading bot
##################################################################################

class bot():

    def __init__(self):

        self.shares_on_hand = 0.
        self.cash_on_hand = seed_money
        self.commission = commission
        self.fixed_rate = fixed_rate
        self.fixed_share = fixed_share


    def buy_share(self, current_price):

        cash_needed = current_price + self.commission
        shares_to_buy = self.fixed_share

        if self.cash_on_hand > cash_needed:
            self.cash_on_hand -= cash_needed
            self.shares_on_hand += shares_to_buy



    def buy_fixed(self, current_price):

        cash_needed = self.fixed_rate + self.commission
        shares_to_buy = self.fixed_rate / current_price

        if self.cash_on_hand > cash_needed:        
            self.cash_on_hand -= cash_needed
            self.shares_on_hand += shares_to_buy



    def sell_share(self, current_price):

        shares_needed = self.fixed_share
        cash_gain = current_price - self.commission
       
        if self.shares_on_hand > 1:
            self.cash_on_hand += cash_gain
            self.shares_on_hand -= shares_needed



    def sell_fixed(self, current_price):

        shares_needed = (self.fixed_rate + self.commission) / current_price
        cash_gain = self.fixed_rate 
       
        if self.shares_on_hand > shares_needed:
            self.cash_on_hand += self.fixed_rate
            self.shares_on_hand -= shares_needed


    def cash_out(self, current_price):

        cash_from_sale = self.shares_on_hand * current_price - self.commission

        self.shares_on_hand = 0
        self.cash_on_hand += cash_from_sale


##########################################################################
# data is from  https://www.measuringworth.com/datasets/DJA/index.php
##########################################################################
# read in DJA
dja = pd.read_csv('DJA.csv')        # 31747 days of data 
n_samples = len(dja)


# set share price to be 1/1000 of dja
dja['SharePrice'] = dja['DJIA'] / 1000.


# calculate moving avg, crossovers and over/under
dja['MA'] = dja['DJIA'].rolling(window=250).mean()
dja['OverUnder'] = np.where(dja['DJIA'] > dja['MA'], 1, 0)
dja['CrossOver'] = np.where(dja['OverUnder'] != dja['OverUnder'].shift(1), 1, 0)


# look at data, make sure looks good.
#print(dja[['Date', 'DJIA','MA', 'OverUnder', 'CrossOver']])


# chose a time frame to run simulations
dja['year'] = pd.DatetimeIndex(dja.Date).year
dja = dja[dja.year >= start_year]
dja = dja[dja.year <= end_year]


bot_shares = bot()
bot_fixed = bot()


# make list of cross over days and split into over/under days
crossOverDays = dja[dja['CrossOver'] == 1]

for ix, row in crossOverDays.iterrows():

    if row['OverUnder'] == 1:       # buy
        bot_shares.buy_share(row['SharePrice'])
        bot_fixed.buy_fixed(row['SharePrice'])
        print("Buy")
        print("Bot shares", bot_shares.shares_on_hand, bot_shares.cash_on_hand)
        print("Bot fixed", bot_fixed.shares_on_hand, bot_fixed.cash_on_hand)
    else:                           # sell
        bot_shares.sell_share(row['SharePrice'])
        bot_fixed.sell_fixed(row['SharePrice'])
        print("Sell")
        print("Bot shares", bot_shares.shares_on_hand, bot_shares.cash_on_hand)
        print("Bot fixed", bot_fixed.shares_on_hand, bot_fixed.cash_on_hand)




# cash out
bot_shares.cash_out(crossOverDays.iloc[-1]['SharePrice'])
bot_fixed.cash_out(crossOverDays.iloc[-1]['SharePrice'])

print("Final tally:")
print("Shares bot final $", bot_shares.cash_on_hand )
print("Fixed bot final $", bot_fixed.cash_on_hand)


# print ma days
print("Cross over days:")
print(crossOverDays[['Date', 'DJIA', 'SharePrice', 'MA', 'OverUnder']])








# write to disk
dja.to_csv("movingAvg_dja.csv")

