# http://github.com/timestocome


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from fbprophet import Prophet


# https://research.fb.com/prophet-forecasting-at-scale/
# https://facebookincubator.github.io/prophet/docs/quick_start.html

###################################################################################
# read in and set up data
# read in nasdaq 
data = pd.read_csv('data/nasdaq.csv', parse_dates=True, index_col='Date')
data['Open'] = pd.to_numeric(data['Open'], errors='coerce')

# convert to log scale
data['LogOpen'] = np.log(data[['Open']]) 

# fbProphet expects a date series with the data column named 'y', and the date named 'ds'
df = pd.DataFrame({'ds':data.index, 'y':data.LogOpen}) 

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
plt.title("Nasdaq forecast using FBProphet")
plt.savefig("Prophet_Nasdaq_forecast.png")
plt.show()


m.plot_components(forecast)
plt.title('Components of FBProphet Nasdaq Forecast')
plt.savefig('Prophet_Nasdaq_components.png')
plt.show()