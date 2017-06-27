# http://github.com/timestocome

# Attempt to predict gold prices and find outliers


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from fbprophet import Prophet

# https://research.fb.com/prophet-forecasting-at-scale/
# https://facebookincubator.github.io/prophet/docs/quick_start.html
# https://facebookincubator.github.io/prophet/static/prophet_paper_20170113.pdf

######################################################################
# load data
########################################################################
# read in gold file
data = pd.read_csv('data/Gold_all.csv', parse_dates=True, index_col=0)
data['Open'] = pd.to_numeric(data['Open'], errors='coerce')


# split into pre/post US Gold Standard
gold_standard = data.loc[data.index < '01-01-1971']
gold = data.loc[data.index > '01-01-1971']
#print(len(gold_standard), len(gold))

data['LogOpen'] = np.log(data[['Open']])


# fbProphet expects a date series with the data column named 'y', and the date named 'ds'
df = pd.DataFrame({'ds':data.index, 'y':data.LogOpen}) 
df = df.loc[df.index >= '01-01-1971']

# check everything looks okay
print(df)


###################################################################################
# model 

# create and fit model
m = Prophet()
m.fit(df)

# prediction 
future = m.make_future_dataframe(periods=251)       # 251 ~one year of trading days
forecast = m.predict(future)
#print(forecast)

m.plot(forecast)
plt.title("Gold forecast using FBProphet")
plt.savefig("Prophet_Gold_forecast.png")
plt.show()


m.plot_components(forecast)
plt.title('Components of FBProphet Gold Forecast')
plt.savefig('Prophet_Gold_components.png')
plt.show()
