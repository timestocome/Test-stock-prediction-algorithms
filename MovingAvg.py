# http://github.com/timestocome



# test buy when MA crosses under share price, sell when share price over MA
# 

# First run, brute force through all moving average windows
# using either fixed share amounts or fixed dollar amounts
# buy under, sell over
# Buying and selling on MA won't work unless you invest at least $500, otherwise commissions
# eat up all of your profit
#
# At $500+/trade you can make money but the longer your window the more you make so buy and hold is
# still the best strategy.
# Purchasing a fixed amount of shares each year on the first trading day and holding earns about 
# double what buying and selling the MA does
# MA best window is 186 days
# Worst were below 60 and between 237-247 ??? I plotted commissions, there's not a big jump at either window





# todo:
# create buy set amount each year and hold for comparison
#
# try RL bots and see how ML does
# try GA and see how bots do
#
# plot best worst strategies 
# print monthly, yearly gains and losses for best and worst strategies




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 



print("Test buying / selling on moving average with $10,000 seed money")
print("Test fixed dollar purchases between 1980-2016")



# select years
start_year = 1980           # data is 1900-2017 pick year to start training 
end_year = 2016

seed_money = 10000.         # starting cash for bots
commission = 7.             # flat commission per trade
fixed_rate = 500.           # fixed dollar bot 








###################################################################################
# trading bot
##################################################################################

class bot():

    def __init__(self):

        self.shares_on_hand = 0.
        self.cash_on_hand = seed_money
        self.commission = commission
        self.fixed_rate = fixed_rate


  
    def buy_fixed(self, current_price):

        cash_needed = self.fixed_rate + self.commission
        shares_to_buy = self.fixed_rate / current_price

        if self.cash_on_hand > cash_needed:        
            self.cash_on_hand -= cash_needed
            self.shares_on_hand += shares_to_buy



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
dja["Year"] = pd.DatetimeIndex(dja['Date']).year
dja['SharePrice'] = dja['DJIA'] / 1000.



##########################################################################
# loop over several MA windows
##########################################################################

smallest_window = 21     # ~1 month
largest_window = 251    # ~ 1 year
windows = range(smallest_window, largest_window)


# for plotting
profit = []         # cash on hand at end
fees = []


for w in windows:
    
    # calculate moving avg, crossovers and over/under
    dja['MA'] = dja['DJIA'].rolling(window=w).mean()
    dja['OverUnder'] = np.where(dja['DJIA'] > dja['MA'], 1, 0)
    dja['CrossOver'] = np.where(dja['OverUnder'] != dja['OverUnder'].shift(1), 1, 0)


    # look at data, make sure looks good.
    #print(dja[['Date', 'DJIA','MA', 'OverUnder', 'CrossOver']])


    # chose a time frame to run simulations
    dja['year'] = pd.DatetimeIndex(dja.Date).year
    dja = dja[dja.year >= start_year]
    dja = dja[dja.year <= end_year]


    trader = bot()
    

    # make list of cross over days and split into over/under days
    crossOverDays = dja[dja['CrossOver'] == 1]
    last_price = 0

    for ix, row in crossOverDays.iterrows():

        if row['OverUnder'] == 1:       # buy
            trader.buy_fixed(row['SharePrice'])
        else:                           # sell
            trader.sell_fixed(row['SharePrice'])
        
        last_price = row['SharePrice']



    # cash out
    trader.cash_out(crossOverDays.iloc[-1]['SharePrice'])



    

    print("*************************************************************")
    n_crossover_days = len(crossOverDays)
    print("Final $ %.2lf using %d trading days as window:" % (trader.cash_on_hand, w))
    print("Cross over days: %d, trading fees $%.2lf" % (n_crossover_days, n_crossover_days * commission))



    # save for plotting
    fees.append(n_crossover_days * commission)
    profit.append(trader.cash_on_hand)



########################################################################
# end loop
#######################################################################

######################################################################
# create buy a set a month first trading day each year to compare
####################################################################
years = end_year - start_year
yearly_investment_dollars = seed_money / years

one_trade_yr = dja.groupby(['Year']).first()

# chose a time frame to run simulations
one_trade_yr = one_trade_yr[one_trade_yr.year >= start_year]
one_trade_yr = one_trade_yr[one_trade_yr.year <= end_year]


total_shares_buy_and_hold = 0
for ix, row in one_trade_yr.iterrows():
    new_shares = (yearly_investment_dollars - commission) / row['SharePrice']
    total_shares_buy_and_hold += new_shares


total_buy_and_hold = last_price * total_shares_buy_and_hold - commission
print("Total $%.2lf for buy fixed amount each year " % total_buy_and_hold)


plt.title("Trade on Moving Average 1980-2016, Start with 10k seed money, $500 trades")
plt.plot(profit, c='b', linewidth=3)
plt.plot(fees, c='r', linewidth=2)
plt.show()



