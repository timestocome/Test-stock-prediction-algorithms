# http://github.com/timestocome


# Attempt to use velocity, acceleration, momentum, energy, force, hooke's law
# to predict changes 1 week, month, quarter into the future
# This regression only uses data from the index itself


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf 


# pandas display options
pd.options.display.max_rows = 100
pd.options.display.max_columns = 25
pd.options.display.width = 1000


one_day = 1
one_week = 5
one_month = 21
one_quarter = 63


learning_rate = 0.01
training_epochs = 2



# read in file
def read_data(file_name):

    stock = pd.read_csv(file_name, parse_dates=True, index_col=0)        # 31747 days of data 
    n_samples = len(stock)

    
    # want to predict future gain/loss, these are the target values using in training the model
    stock['next_week'] = np.log( stock['Close'] / stock['Close'].shift(one_week) )
    stock['next_month'] = np.log( stock['Close'] / stock['Close'].shift(one_month) )
    stock['next_quarter'] = np.log( stock['Close'] / stock['Close'].shift(one_quarter) )
    

    # scale
    stock['Open'] = (stock['Open'] - stock['Open'].min()) / stock['Open'].max()

   
    # add in useful things
    stock['Velocity'] = stock['Open'] - stock['Open'].shift(1)
    stock['Acceleration'] = stock['Velocity'] - stock['Velocity'].shift(1)
    stock['Momentum'] = stock['Open'] * stock['Velocity']
    stock['Energy'] = stock['Open'] * stock['Velocity'] * stock['Velocity']
    stock['Force'] = stock['Open'] * stock['Acceleration']
    stock['Elastic'] = stock['Open'] * stock['Open']

    stock['VelocityAbs'] = stock['Velocity'].abs()
    stock['AccelerationAbs'] = stock['Acceleration'].abs()
    stock['MomentumAbs'] = stock['Momentum'].abs()
    stock['EnergyAbs'] = stock['Energy'].abs()
    stock['ForceAbs'] = stock['Force'].abs()
    stock['ElasticAbs'] = stock['Elastic'].abs()


    # scale volume
    stock['Volume'] = (stock['Volume'] - stock['Volume'].min()) / stock['Volume'].max()


    # ditch samples with NAN values
    stock = stock.dropna(axis=0)


    # flip order from newest to oldest to oldest to newest
    stock = stock.iloc[::-1]

    # shuffle data
    #stock = stock.sample(frac=1)


    # split data set into training and holdout
    # hold out all dates > 1/1/2017
    hold_out_stock = stock.loc[stock.index > '01-01-2016']

    # test and train on 1/1/85-12/31/2016
    train_stock = stock.loc[stock.index > '01-01-1985']
    train_stock = stock.loc[stock.index < '12-31-2015']



    # all stock is needed to walk back dates for testing hold out data
    return stock, train_stock, hold_out_stock



#############################################################################################
#############################################################################################
# split into train, test, predict

print("Training scores:")

print("DJIA")
dj_stock, dj_train, dj_hold_out = read_data('data/djia.csv')

print("S&P")
sp_stock, sp_train, sp_hold_out = read_data('data/S&P.csv')

print("Russell")
r_stock, r_train, r_hold_out = read_data('data/Russell2000.csv')

print("NASDAQ")
n_stock, n_train, n_hold_out = read_data('data/nasdaq.csv')

#############################################################################################
# check correlations
def check_features():
    print("****************   Correlations ********************************")

    features = ['Open', 'Volume', 'next_week', 'next_month', 'next_quarter', 'Velocity', 'Momentum', 'Energy', 'Elastic', 'VelocityAbs', 'AccelerationAbs', 'MomentumAbs', 'EnergyAbs', 'ForceAbs', 'ElasticAbs']

    #features = ['Open', 'Volume', 'Energy', 'Elastic', 'VelocityAbs', 'AccelerationAbs', 'MomentumAbs', 'EnergyAbs', 'ForceAbs', 'ElasticAbs']
    correlations = dj_stock[features].corr()

    print(correlations[['Open', 'next_week', 'next_month', 'next_quarter']])



# ditch features that don't effect the future prices
check_features()








# use correlations function below to find good features
features = ['Open', 'Volume', 'Velocity', 'Acceleration', 'Momentum', 'Energy', 'Force', 'Elastic', 'VelocityAbs', 'AccelerationAbs', 'MomentumAbs', 'EnergyAbs', 'ForceAbs', 'ElasticAbs']
features = ['Open', 'Elastic', 'AccelerationAbs', 'Volume']

target = ['next_quarter']       # next_week, next_month, next_quarter

# convert current training set to numpy array
x_train = dj_train.as_matrix(columns=[features])
y_train = dj_train.as_matrix(columns=[target])
n_features = len(features)
n_out = 1
n_samples = len(y_train)

X = tf.placeholder('float')
Y = tf.placeholder('float')

def model(X, w):
    return tf.multiply(X, w)

w = tf.Variable(tf.random_normal([n_features, n_out]),  name='weights')
y_model = model(X, w)
cost = tf.square(Y - y_model)
predict = tf.nn.relu(tf.multiply(w, X))

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


for epoch in range(training_epochs):
    for (x, y) in zip(x_train, y_train):
        sess.run(train_op, feed_dict={X: x, Y: y})


w_value = sess.run(w)

sess.close()


################################################
# plot training data
################################################
plt.figure(figsize=(15,15))

plt.suptitle("Test Stock Predictions")

ax1 = plt.subplot(211)
ax1.plot(y_train, 'b', label='Actual')

y_learned = np.empty([n_samples])
i = 0
for x in x_train:

    y_learned[i] = (x * w_value).sum()
    i += 1

ax1.plot(y_learned, 'r', label='Predicted')
ax1.set_title("Training data predictions")
ax1.legend(loc='best')

#############################################
# plot hold out data
##############################################
x_test = dj_hold_out.as_matrix(columns=[features])
y_test = dj_hold_out.as_matrix(columns=[target])

z = range(len(y_learned), len(y_learned)+len(y_test))

ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
ax2.plot(z, y_test, 'b', label='Actual')

y_predicted = np.empty([len(y_test)])
i = 0
for x in x_test:
    y_predicted[i] = (x * w_value).sum()
    i += 1

ax2.plot(z, y_predicted, 'r', label='Prediction')

ax2.set_title("Hold out data predictions")
ax2.legend(loc='best')

plt.savefig('LinearRegression_Change.png')

plt.show()

print(len(dj_hold_out))


