# http://github.com/timestocome


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


# pandas display options
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 25
pd.options.display.width = 1000


# https://www.cs.elte.hu/blobs/diplomamunkak/bsc_matelem/2014/fora_gyula_krisztian.pdf
# only used a couple of the 70 odd technical indicators
# and I used different economic indicators
# still gets about 70% accuracy predicting whether Nasdaq will rise or fall tomorrow



# read in nasdaq 
data = pd.read_csv('StockData.csv', parse_dates=True, index_col='Date')

# cast strings to floats
data['Russell'] = np.log(pd.to_numeric(data['Russell 2000'], errors='coerce'))
data['Nasdaq'] = np.log( pd.to_numeric(data['NASDAQ'], errors='coerce'))
data['Gold'] = np.log(pd.to_numeric(data['Gold'], errors='coerce'))
data['1yr TBill'] = np.log(pd.to_numeric(data['1yr Treasury'], errors='coerce'))

data = data[['Russell', 'Nasdaq', 'Gold', '1yr TBill']]



# index change from prior day
data['dx'] = data['Nasdaq'] - data['Nasdaq'].shift(1)
data = data.dropna()

# check every thing looks okay
#print(data)


#################################################################
# histogram of returns 

'''
n_bins = 100
plt.figure(figsize=(8, 8))
plt.title("Daily returns on Nasdaq")
plt.hist(data['dx'], n_bins, histtype='bar', normed=1)
plt.show()
'''

########################################################################
# technical indicators as features


window = 251
data['ema'] = data['Nasdaq'].ewm(span=window).mean()

data['macd'] = data['ema'] - data['Nasdaq'].ewm(span=window/2).mean()

data['rolling_min'] = data['Nasdaq'].rolling(window=window).min()
data['rolling_max'] = data['Nasdaq'].rolling(window=window).max()
data['K'] = (data['Nasdaq'] - data['rolling_min'])/(data['rolling_max'] - data['rolling_min']) 




#########################################################################
# gold as an economic uncertainty index
# russell 2000 is supposed to lead other indexes
# 1 yr treasury

# author uses change not absolute values
# 'Russell', 'Nasdaq', 'Gold', '1yr TBill'
data['Russell'] = data['Russell'] - data['Russell'].shift(1)
data['Gold'] = data['Gold'] - data['Gold'].shift(1)
data['1yr TBill'] = data['1yr TBill'] - data['1yr TBill'].shift(1)

data = data.dropna()
#print(data)

# which features are coorelated with price movement (dx )?
print("Feature Correlation with Nasdaq daily price changes")
print(data.corr()['dx'])

#######################################################
# model
from sklearn.ensemble import RandomForestClassifier


data['y'] = np.where(data['dx'] > 0, 1, -1)

train = data.loc[data.index <= '12-31-2015']
test = data.loc[data.index > '12-31-2015']

x = train[['ema', 'macd', 'K', '1yr TBill', 'Gold', 'Russell']]
y = train['y']

test_x = test[['ema', 'macd', 'K', '1yr TBill', 'Gold', 'Russell']]
test_y = test['y']

print(len(x), len(test_x))

model = RandomForestClassifier()
model = model.fit(x, y)
score = model.score(test_x, test_y)

print("Test model on tomorrow's gain/loss")
print("Accuracy: ", score)