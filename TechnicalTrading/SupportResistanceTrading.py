# http://github.com/timestocome


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


# pandas display options
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 25
pd.options.display.width = 1000


# taking a dive into technical analysis
# http://www.mrao.cam.ac.uk/~mph/Technical_Analysis.pdf


# read in nasdaq 
data = pd.read_csv('StockData.csv', parse_dates=True, index_col='Date')

# cast strings to floats
data['Russell'] = np.log(pd.to_numeric(data['Russell 2000'], errors='coerce'))
data['Nasdaq'] = np.log( pd.to_numeric(data['NASDAQ'], errors='coerce'))
data['Gold'] = np.log(pd.to_numeric(data['Gold'], errors='coerce'))
data['1yr TBill'] = np.log(pd.to_numeric(data['1yr Treasury'], errors='coerce'))

data = data[['Russell', 'Nasdaq', 'Gold', '1yr TBill']]



#################################################################
# support levels
window = 63        # number of trading days to use for trend lines

# find highest and lowest points inside window
data['Gold_min'] = data['Gold'].rolling(window=window).min()
data['Nasdaq_min'] = data['Nasdaq'].rolling(window=window).min()

data['Gold_max'] = data['Gold'].rolling(window=window).max()
data['Nasdaq_max'] = data['Nasdaq'].rolling(window=window).max()



##########################################################################
# trade using support levels

commission = 7.     # price per buy/sell
beginning_balance = 1000.
balance_gold = beginning_balance
balance_nasdaq = beginning_balance
ounces_gold = 0
shares_nasdaq = 0

# start with begining balance $1000
# buy and trade all on hand on signal
for idx, row in data.iterrows():

    if row.Gold >= row.Gold_max:        # sell gold
        if ounces_gold > 0.:
            balance_gold -= commission
            balance_gold += ounces_gold * row.Gold 
            ounces_gold = 0.

    if row.Gold <= row.Gold_min:         # buy gold
        if balance_gold > 0.:
            balance_gold -= commission
            ounces_gold += balance_gold / row.Gold 
            balance_gold = 0
    
    if row.Nasdaq >= row.Nasdaq_max:    # sell nasdaq
        if shares_nasdaq > 0.:
            balance_nasdaq -= commission
            balance_nasdaq += shares_nasdaq * row.Nasdaq
            shares_nasdaq = 0.
    
    if row.Nasdaq <= row.Nasdaq_min:       # buy nasdaq
        if balance_nasdaq > 0.: 
            balance_nasdaq -= commission
            shares_nasdaq += balance_nasdaq / row.Nasdaq
            balance_nasdaq = 0.

    

# cash out at end
last_gold_price = data.iloc[-1].Gold 
last_nasdaq_price = data.iloc[-1].Nasdaq 

# if still holding shares/gold cash out
if ounces_gold > 0.:
    balance_gold -= commission
    balance_gold += last_gold_price * ounces_gold

if shares_nasdaq > 0.:
    balance_nasdaq -= commission
    balance_nasdaq += last_nasdaq_price * shares_nasdaq



nasdaq_gain = balance_nasdaq - beginning_balance 
gold_gain = balance_gold - beginning_balance



####################################################################
# plots

t = 'Support level trading using ' + str(window) + ' trading days, gold gain/loss: ' + str(gold_gain) + ' Nasdaq gain/loss: ' + str(nasdaq_gain)


plt.figure(figsize=(16,16))
plt.title(t)
plt.plot(data['Gold_max'], lw=3, alpha=0.4, color='salmon')
plt.plot(data['Gold_min'], lw=3, alpha=0.4, color='darkseagreen')
plt.plot(data['Gold'], label='Gold')
plt.plot(data['Nasdaq_max'], lw=3, alpha=0.4, color='salmon')
plt.plot(data['Nasdaq_min'], lw=3, alpha=0.4, color='darkseagreen')
plt.plot(data['Nasdaq'], label='Nasdaq')
plt.legend(loc='best')
plt.grid('on')
plt.savefig('SupportResistanceTrading.png')
plt.show()
