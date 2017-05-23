# http://github.com/timestocome

# take a look at the differences in daily returns for recent bull and bear markets
# http://afoysal.blogspot.com/2016/08/arma-and-arima-timeseries-prediction.html


# predictions appear to increase and decrease with actual returns but scale is much smaller
# of course if it was this easy there'd be a lot of rich statisticians in the world.

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic
import matplotlib.pyplot as plt 


# pandas display options
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 25
pd.options.display.width = 1000

######################################################################
# data
########################################################################

# read in datafile created in LoadAndMatchDates.py
data = pd.read_csv('StockDataWithVolume.csv', index_col='Date', parse_dates=True)
features = [data.columns.values]


# create target --- let's try Nasdaq value 1 day change
data['returns'] = (data['NASDAQ'] - data['NASDAQ'].shift(1)) / data['NASDAQ']


# remove nan row from target creation
data = data.dropna()


#########################################################################
# split into bear and bull markets
##########################################################################

bull1_start = pd.to_datetime('01-01-1990')       # beginning of this dataset
bull1_end = pd.to_datetime('07-16-1990')

iraq_bear_start = pd.to_datetime('07-17-1990')
iraq_bear_end = pd.to_datetime('10-11-1990')

bull2_start = pd.to_datetime('10-12-1990')
bull2_end = pd.to_datetime('01-13-2000')

dotcom_bear_start = pd.to_datetime('01-14-2000')
dotcom_bear_end = pd.to_datetime('10-09-2002')

bull3_start = pd.to_datetime('10-10-2002')
bull3_end = pd.to_datetime('10-08-2007')

housing_bear_start = pd.to_datetime('10-09-2007')
housing_bear_end = pd.to_datetime('03-09-2009')

bull4_start = pd.to_datetime('03-10-2009')
bull4_end = pd.to_datetime('12-31-2016')    # end of this dataset



bull1 = data.loc[data.index <= bull1_end]
bear1 = data.loc[(data.index >= iraq_bear_start) & (data.index <= iraq_bear_end)]
bull2 = data.loc[(data.index >= bull2_start) & (data.index <= bull2_end)]
bear2 = data.loc[(data.index >= dotcom_bear_start) & (data.index <= dotcom_bear_end)]
bull3 = data.loc[(data.index >= bull3_start) & (data.index <= bull3_end)]
bear3 = data.loc[(data.index >= housing_bear_start) & (data.index <= housing_bear_end)]
bull4 = data.loc[data.index >= bull4_start]

#######################################################################
# stats
#######################################################################
'''
plt.figure(figsize=(12,12))
data.returns.plot.line(style='darkgreen', legend=True, grid=True, label='DailyReturns')
ax = data.returns.rolling(window=21).mean().plot.line(style='darkseagreen', legend=True, label="Monthly")
ax.set_xlabel("Date")
plt.legend(loc='best')
plt.title('Nasdaq')
plt.show()
'''

# check data is stationary ( statistical values do not change over time )
#
def stationary_test(df):
    print('Results of Dickey-Fuller Test:')
    df_test=adfuller(df)
    indices = ['Test Statistic', 'p-value', 'No. Lags Used', 'Number of Observations Used']
    output = pd.Series(df_test[0:4], index=indices) # 4 lags for quarterly, 12 for monthly
    for key, value in df_test[4].items():
        output['Critical value (%s)' % key] = value
    print(output)

stationary_test(data['returns'])


# looks okay, p-value is close to zero
# ARMA Auto regressive moving average
# p is the autoregressive count
# q is the moving average count
# let pandas select best p, q
#print (arma_order_select_ic(data.returns, ic=['aic', 'bic'], trend='nc',
#                            max_ar=5, max_ma=5, fit_kw={'method': 'css-mle'}))

# returns 5,2 and 0, 1
ts = data.returns
model = ARMA(ts, order=(5,2))
results = model.fit(trend='nc', method='css-mle', disp=-1)
print(results.summary2())
predicted = results.predict('01-03-1990', '12-30-2016')


plt.figure(figsize=(16, 16))
plt.title("Nasdaq actual returns vs ARMA predicted")
plt.plot(data['returns'], c='blue', label='Actual')
plt.plot(predicted, c='darkseagreen', label='Predicted')
plt.legend(loc='best')
plt.show()

################################################################

model_arima = ARIMA(ts, order=(5, 1, 2))
results_arima = model_arima.fit(disp=-1, transparams=True)
print(results_arima.summary2())
predicted = results.predict('01-03-1990', '12-30-2016')



plt.figure(figsize=(16, 16))
plt.title("Nasdaq actual returns vs ARIMA predicted")
plt.plot(data['returns'], c='blue', label='Actual')
plt.plot(predicted, c='darkseagreen', label='Predicted')
plt.legend(loc='best')
plt.savefig("ARIMA.png")
plt.show()






'''
#################################################################
# plots
###############################################################
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
plt.title('Bull markets weekly returns')
plt.plot(bull1['returns'].resample('W').mean())
plt.plot(bull2['returns'].resample('W').mean())
plt.plot(bull3['returns'].resample('W').mean())
plt.plot(bull4['returns'].resample('W').mean())
plt.grid(True)
plt.xlim(pd.to_datetime('01-01-1990'), pd.to_datetime('12-31-2016'))


plt.subplot(2,1,2)
plt.title('Bear markets weekly returns')
plt.plot(bear1['returns'].resample('W').mean())
plt.plot(bear2['returns'].resample('W').mean())
plt.plot(bear3['returns'].resample('W').mean())
plt.grid(True)
plt.xlim(pd.to_datetime('01-01-1990'), pd.to_datetime('12-31-2016'))

plt.show()
'''
