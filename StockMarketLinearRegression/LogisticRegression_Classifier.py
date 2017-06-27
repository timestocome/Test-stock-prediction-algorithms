# http://github.com/timestocome

# classifier in tf using linear regression
# meh, best cost is still about 22% error rate predicting if Nasdaq will go up or down tomorrow
# when using VIX, GDP % chg, Gold, 1yr Treas, 10 Bond, GDP actual, Unemployment rate


import pandas as pd 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



# pandas display options
pd.options.display.max_rows = 100
pd.options.display.max_columns = 25
pd.options.display.width = 1000

# read in datafile created in LoadAndMatchDates.py
data = pd.read_csv('StockData.csv', index_col='Date', parse_dates=True)


# add daily changes for dj, sp, russell, nasdaq, vix, gold, 
data['djia_dx'] = (data['DJIA'] - data['DJIA'].shift(1)) / data['DJIA']
data['sp_dx'] = (data['S&P'] - data['S&P'].shift(1)) / data['S&P']
data['russell_dx'] = (data['Russell 2000'] - data['Russell 2000'].shift(1)) / data['Russell 2000']
data['nasdaq_dx'] = (data['NASDAQ'] - data['NASDAQ'].shift(1)) / data['NASDAQ']


# convert daily changes into 0 and 1 for decreasing or increasing change in marktet
data['djia_target'] = data['djia_dx'].apply(lambda z: 1 if z > 0 else 0)
data['sp_target'] = data['sp_dx'].apply(lambda z: 1 if z > 0 else 0)
data['russell_target'] = data['russell_dx'].apply(lambda z: 1 if z > 0 else 0)
data['nasdaq_target'] = data['nasdaq_dx'].apply(lambda z: 1 if z > 0 else 0)



# downsample from days to weeks, now have ~1400 data samples
weekly_data = data.resample('W-SUN').sum()


# scale data
scale_max = 1
scale_min = -1
weekly_data = (weekly_data - weekly_data.min()) / (weekly_data.max() - weekly_data.min()) * (scale_max - scale_min) + scale_min




#############################################################################
# peak NASDAQ gain/losses in a week
#############################################################################
n_peaks = 50
top_gains = weekly_data.nlargest(n_peaks, 'nasdaq_dx')
top_losses = weekly_data.nsmallest(n_peaks, 'nasdaq_dx')


'''
print("top gains --------------------------------------------------------------------------------")
print(top_gains[['nasdaq_dx']])
print("top losses -------------------------------------------------------------------------------")
print(top_losses[['nasdaq_dx']])
'''

#############################################################################################################
# classify winning weeks from losing weeks
#########################################################################################################

features = ['DJIA', 'S&P', 'Russell 2000', 'NASDAQ', 'VIX', 'US GDP', 'Gold', '1yr Treasury', '10yr Bond', 'Real GDP', 'UnEmploy', 'djia_dx', 'sp_dx', 'russell_dx', 'nasdaq_dx']

print("----------------------------------------------------------")
print("Correlation Matrix")
correlation_matrix = weekly_data.corr()
print(correlation_matrix['nasdaq_dx'])

print("----------------------------------------------------------")
print("Coveriance Matrix")
coveriance_matrix = weekly_data.cov()
print(coveriance_matrix['nasdaq_dx'])


# features to keep to predict NASDAQ change
features = ['VIX', 'US GDP', 'Gold', '1yr Treasury','10yr Bond', 'Real GDP', 'UnEmploy']
x = (weekly_data[features]).as_matrix()
y = (weekly_data['nasdaq_target']).as_matrix()



##############################################################################################
# TF model classification using linear regression
#############################################################################################

learning_rate = 0.001
epochs = 20
n_samples = len(x)

X = tf.placeholder('float')
Y = tf.placeholder('float')
w = tf.Variable([.1] * len(features), name='parameters')


def model(X, w):
    return tf.multiply(w, X)
    

y_predicted = model(X, w)
cost = tf.reduce_mean(tf.square(Y - y_predicted))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print("----------------------------------------------------------")
print("Training")

for epoch in range(epochs):
    for (d, t) in zip(x, y):
        sess.run(train_op, {X: d, Y:t})
        current_p = sess.run(y_predicted, {X:d, Y:t})
    current_cost = sess.run(cost, {X: d, Y: t})
    print("Epoch: %d, current_cost: %lf" % (epoch, current_cost))

print("----------------------------------------------------------")
print("Weights")
w_values = sess.run(w)
for i in range(len(features)):
    print("Feature: %s Weights: %lf" % (features[i], w_values[i]))
sess.close()
