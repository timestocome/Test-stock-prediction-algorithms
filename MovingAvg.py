# http://github.com/timestocome



# test buy when MA crosses under share price, sell when share price over MA


# Simple test brute force through all moving average windows
# using fixed dollar amounts to buy and sell.
#
# Buying and selling on MA won't work unless you invest at least $500, otherwise commissions
# eat up all of your profit
#
#
# Using $500/purchase and sell at each moving average crossover
# DJIA
# Buy a set amount 1 trading day of each year earns about $56k
# Best buy/sell on moving average is 186 day window and earns $23k
# Worst were below 60 and between 237-247 ??? I plotted commissions, there's not a big jump at either window
# I have no idea yet why using a MA window between 237-247 is so bad.
#
# NASDAQ
# Buy a set amount the first trading day of the year earns about $3k
# Using an 80-90 moving average nets you about $7K
#
# S&P 500
# Set amount each year yeilds about $3k
# MA 224-250 yeilds $7k





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 



print("Test buying / selling on moving average with $10,000 seed money")
print("Test fixed dollar purchases between 1980-2016")



# select years
start_year = 1980           # data is 1900-2017 pick year to start training 
end_year = 2016


# bot stuff
seed_money = 10000.         # starting cash for bots
commission = 7.             # flat commission per trade
fixed_rate = 500.           # fixed dollar bot 


# moving average window sized to loop over
smallest_window = 21     # ~1 month
largest_window = 251    # ~ 1 year





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
# and http://finance.yahoo.com 
##########################################################################
'''
# read in DJA
stock_data = pd.read_csv('DJA.csv')        # 31747 days of data 
n_samples = len(stock_data)

# set share price to be 1/1000 of dja
stock_data["Year"] = pd.DatetimeIndex(stock_data['Date']).year
stock_data['SharePrice'] = stock_data['DJIA'] / 1000.

stock_data['Close'] = stock_data['DJIA']

'''

'''
# read in NASDAQ
stock_data = pd.read_csv('nasdaq.csv')
stock_data = stock_data[['Date', 'Close']]


# set share price to be 1/1000 of daily index
stock_data["Year"] = pd.DatetimeIndex(stock_data['Date']).year
stock_data['SharePrice'] = stock_data['Close'] / 1000.
'''


# read in S&P
stock_data = pd.read_csv('S&P.csv')
stock_data = stock_data[['Date', 'Close']]


# set share price to be 1/1000 of daily index
stock_data["Year"] = pd.DatetimeIndex(stock_data['Date']).year
stock_data['SharePrice'] = stock_data['Close'] / 1000.






##########################################################################
# loop over several MA windows
##########################################################################


windows = range(smallest_window, largest_window)


# for plotting
profit = []         # cash on hand at end
fees = []           # trading fees ( commissions )
window = []         # days in trading window


for w in windows:
    
    # calculate moving avg, crossovers and over/under
    stock_data['MA'] = stock_data['Close'].rolling(window=w).mean()
    stock_data['OverUnder'] = np.where(stock_data['Close'] > stock_data['MA'], 1, 0)
    stock_data['CrossOver'] = np.where(stock_data['OverUnder'] != stock_data['OverUnder'].shift(1), 1, 0)


    # look at data, make sure looks good.
    #print(dja[['Date', 'DJIA','MA', 'OverUnder', 'CrossOver']])


    # set the time frame to run simulations
    stock_data['year'] = pd.DatetimeIndex(stock_data.Date).year
    stock_data = stock_data[stock_data.year >= start_year]
    stock_data = stock_data[stock_data.year <= end_year]


    trader = bot()
    window.append(w)
    

    # make list of cross over days and split into over/under days
    crossOverDays = stock_data[stock_data['CrossOver'] == 1]
    last_price = 0

    for ix, row in crossOverDays.iterrows():

        if row['OverUnder'] == 1:       # buy
            trader.buy_fixed(row['SharePrice'])
        else:                           # sell
            trader.sell_fixed(row['SharePrice'])
        
        last_price = row['SharePrice']  # use to calculate buy and hold cash out amount


    # cash out moving average bot
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

# top windows
indexes = np.argmax(np.asarray(profit))
print("Max: ", indexes)
#for i in indexes:
#    print("Top MA windows: %d days, %.2lf net" % (window[i], profit[i]))


######################################################################
# compare buying set amount 1st of each year to MA bot results
####################################################################
years = end_year - start_year
yearly_investment_dollars = seed_money / years

one_trade_yr = stock_data.groupby(['Year']).first()

# chose a time frame to run simulations
one_trade_yr = one_trade_yr[one_trade_yr.year >= start_year]
one_trade_yr = one_trade_yr[one_trade_yr.year <= end_year]


total_shares_buy_and_hold = 0
for ix, row in one_trade_yr.iterrows():
    new_shares = (yearly_investment_dollars - commission) / row['SharePrice']
    total_shares_buy_and_hold += new_shares


total_buy_and_hold = last_price * total_shares_buy_and_hold - commission
print("Total $%.2lf if bought fixed amount ( 10k/yrs ~ $278 ) each year " % total_buy_and_hold)



##########################################################################
# plot
##########################################################################
plt.title("Trade on S&P Moving Average 1980-2016, Start with 10k seed money, $500 trades")
returns = plt.plot(profit, c='b', linewidth=3, label="Return")
fees = plt.plot(fees, c='r', linewidth=2, label="Fees")
plt.xlabel("Days in moving average window")
plt.ylabel("Total return $")
plt.legend(loc='upper left')
plt.text(y=20, x=3, s="Buying fixed 10k/yrs each year and holding nets you $2951K" )

plt.savefig("MovingAverageTrading.png")

plt.show()

