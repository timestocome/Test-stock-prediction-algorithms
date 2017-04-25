# http://github.com/timestocome


# Attempt to use velocity, acceleration, momentum
# to predict prices 1 week, month, quarter into the future
# test date is 1/9/2017 using data from 1 week, month and quarter ago
# score seems insanely too good to be true. 



import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn import metrics



# pandas display options
pd.options.display.max_rows = 100
pd.options.display.max_columns = 25
pd.options.display.width = 1000


one_day = 1
one_week = 5
one_month = 21
one_quarter = 63



# data to use in training and predictions
features = ['Close', 'Volume', 'dx', 'd2x',  'momentum'] # today's close





# read in file
def read_data(file_name):

    stock = pd.read_csv(file_name, parse_dates=True, index_col=0)        # 31747 days of data 
    n_samples = len(stock)

    
    # want to predict future prices, these are the target values using in training the model
    stock['next_week'] = stock['Close'].shift(one_week)
    stock['next_month'] = stock['Close'].shift(one_month)
    stock['next_quarter'] = stock['Close'].shift(one_quarter)
    

    # calculate velocity, acceleration and momentum
    stock['dx'] = stock['Close'] - stock['Close'].shift(1)
    stock['d2x'] = stock['dx'] - stock['dx'].shift(1)
    stock['momentum'] = ((stock['Close'] * stock['dx'] * stock['dx']) **(1./3.))


    # split data set into training and holdout
    # hold out all dates > 1/1/2017
    hold_out_stock = stock.loc[stock.index > '01-01-2017']

    # test and train on 1/1/85-12/31/2016
    train_stock = stock.loc[stock.index > '01-01-1985']
    train_stock = stock.loc[stock.index < '12-31-2016']

    # all stock is needed to walk back dates for testing hold out data
    return stock, train_stock, hold_out_stock





def train(data):

    
    # one week
    week_model = LinearRegression()
    train, test, target, test_target = train_test_split(data[features], data['next_week'], test_size=0.2)
    week_model = week_model.fit(train, target)
    predictions = week_model.predict(test)
    score = week_model.score(test, test_target)
    print('Accuracy on next week price:', score)

    # one month
    month_model = LinearRegression()
    train, test, target, test_target = train_test_split(data[features], data['next_month'], test_size=0.2)
    month_model = month_model.fit(train, target)
    predictions = month_model.predict(test)
    score = month_model.score(test, test_target)
    print('Accuracy on next month price:', score)

    
    # one quarter
    quarter_model = LinearRegression()
    train, test, target, test_target = train_test_split(data[features], data['next_quarter'], test_size=0.2)
    quarter_model = quarter_model.fit(train, target)
    predictions = quarter_model.predict(test)
    score = quarter_model.score(test, test_target)
    print('Accuracy on next quarter price:', score)

    return week_model, month_model, quarter_model
    





#############################################################################################
# train, test, predict

print("Training scores:")

print("DJIA")
dj_stock, dj_train, dj_hold_out = read_data('data/vma_djia.csv')
dj_week, dj_month, dj_quarter = train(dj_train)

print("S&P")
sp_stock, sp_train, sp_hold_out = read_data('data/vma_S&P.csv')
sp_week, sp_month, sp_quarter = train(sp_train)

print("Russell")
r_stock, r_train, r_hold_out = read_data('data/vma_Russell2000.csv')
r_week, r_month, r_quarter = train(r_train)

print("NASDAQ")
n_stock, n_train, n_hold_out = read_data('data/vma_nasdaq.csv')
n_week, n_month, n_quarter = train(n_train)





print("--------------------------------------------")
print("Test scores:")


################################################################################################
# try a test date outside of training range
# Jan 9
# todo - count back table entries instead of doing it by hand
#####################################################################################################

# features = ['Volume', 'dx', 'percent_dx', 'd2x',  'dx_abs', 'd2x_abs', 'momentum']
# Need features from back dates to use as input to the models

# get a row number
dj_actual = dj_stock.loc[dj_stock.index == '01-09-2017']
row = dj_stock.index.get_loc('01-09-2017')

# Run data through prediction models
predict_w = dj_week.predict((dj_stock.ix[row + one_week])[features])
predict_m = dj_month.predict((dj_stock.ix[row + one_month])[features])
predict_q = dj_quarter.predict((dj_stock.ix[row + one_quarter])[features])
predictions = [predict_m[0], predict_q[0], predict_w[0]]


print("DJIA Actual Jan 9th, 2017: ", dj_actual['Close'][0])
print("Std: ", np.std(predictions))
print("Mean: ", np.mean(predictions))
print("Diff: %.2lf%%" % (np.mean(predictions - dj_actual['Close'][0]) / dj_actual['Close'][0] * 100.))
print("Predictions: ", predictions)



print("--------------------------------------------")

# get a row number
sp_actual = sp_stock.loc[sp_stock.index == '01-09-2017']
row = sp_stock.index.get_loc('01-09-2017')


# Run data through prediction models
predict_w = sp_week.predict((sp_stock.ix[row + one_week])[features])
predict_m = sp_month.predict((sp_stock.ix[row + one_month])[features])
predict_q = sp_quarter.predict((sp_stock.ix[row + one_quarter])[features])
predictions = [predict_w[0], predict_m[0], predict_q[0]]



print("S&P Actual Jan 9th, 2017: ", sp_actual['Close'])
print("Std: ", np.std(predictions))
print("Mean: ", np.mean(predictions))
print("Diff: %.2lf%%" % (np.mean(predictions - sp_actual['Close'][0]) / sp_actual['Close'][0] * 100.))
print("Predictions: ", predictions)

print("--------------------------------------------")

# get a row number
r_actual = r_stock.loc[r_stock.index == '01-09-2017']
row = r_stock.index.get_loc('01-09-2017')


# Run data through prediction models
predict_w = r_week.predict((r_stock.ix[row + one_week])[features])
predict_m = r_month.predict((r_stock.ix[row + one_month])[features])
predict_q = r_quarter.predict((r_stock.ix[row + one_quarter])[features])

predictions = [predict_w[0], predict_m[0], predict_q[0]]


print("Russell 2000 Actual Jan 9th, 2017: ", r_actual['Close'])
print("Std: ", np.std(predictions))
print("Mean: ", np.mean(predictions))
print("Diff: %.2lf%%" % (np.mean(predictions - r_actual['Close'][0]) / r_actual['Close'][0] * 100.))
print("Predictions: ", predictions)

print("--------------------------------------------")

# get a row number
n_actual = n_stock.loc[n_stock.index == '01-09-2017']
row = n_stock.index.get_loc('01-09-2017')

n_actual = n_stock.loc[n_stock.index == '03-29-2017']
row = n_stock.index.get_loc('03-29-2017')

# Run data through prediction models
predict_w = n_week.predict((n_stock.ix[row + one_week])[features])
predict_m = n_month.predict((n_stock.ix[row + one_month])[features])
predict_q = n_quarter.predict((n_stock.ix[row + one_quarter])[features])

predictions = [predict_w[0], predict_m[0], predict_q[0]]

print("NASDAQ Actual Jan 9th, 2017: ", n_actual['Close'])
print("Std: ", np.std(predictions))
print("Mean: ", np.mean(predictions))
print("Diff: %.2lf%%" % (np.mean(predictions - n_actual['Close'][0]) / n_actual['Close'][0] * 100.))
print("Predictions: ", predictions)


# try again
# ---------------------------------------------------------------------------
print("============================================================")
# get a row number
dj_actual = dj_stock.loc[dj_stock.index == '03-29-2017']
row = dj_stock.index.get_loc('03-29-2017')

# Run data through prediction models
predict_w = dj_week.predict((dj_stock.ix[row + one_week])[features])
predict_m = dj_month.predict((dj_stock.ix[row + one_month])[features])
predict_q = dj_quarter.predict((dj_stock.ix[row + one_quarter])[features])
predictions = [predict_m[0], predict_q[0], predict_w[0]]


print("DJIA Actual Mar 29th, 2017: ", dj_actual['Close'][0])
print("Std: ", np.std(predictions))
print("Mean: ", np.mean(predictions))
print("Diff: %.2lf%%" % (np.mean(predictions - dj_actual['Close'][0]) / dj_actual['Close'][0] * 100.))
print("Predictions: ", predictions)



print("--------------------------------------------")

# get a row number
sp_actual = sp_stock.loc[sp_stock.index == '03-29-2017']
row = sp_stock.index.get_loc('03-29-2017')


# Run data through prediction models
predict_w = sp_week.predict((sp_stock.ix[row + one_week])[features])
predict_m = sp_month.predict((sp_stock.ix[row + one_month])[features])
predict_q = sp_quarter.predict((sp_stock.ix[row + one_quarter])[features])
predictions = [predict_w[0], predict_m[0], predict_q[0]]



print("S&P Actual Mar 29th, 2017: ", sp_actual['Close'])
print("Std: ", np.std(predictions))
print("Mean: ", np.mean(predictions))
print("Diff: %.2lf%%" % (np.mean(predictions - sp_actual['Close'][0]) / sp_actual['Close'][0] * 100.))
print("Predictions: ", predictions)

print("--------------------------------------------")

# get a row number
r_actual = r_stock.loc[r_stock.index == '03-29-2017']
row = r_stock.index.get_loc('03-29-2017')


# Run data through prediction models
predict_w = r_week.predict((r_stock.ix[row + one_week])[features])
predict_m = r_month.predict((r_stock.ix[row + one_month])[features])
predict_q = r_quarter.predict((r_stock.ix[row + one_quarter])[features])

predictions = [predict_w[0], predict_m[0], predict_q[0]]


print("Russell 2000 Actual Mar 29th, 2017: ", r_actual['Close'])
print("Std: ", np.std(predictions))
print("Mean: ", np.mean(predictions))
print("Diff: %.2lf%%" % (np.mean(predictions - r_actual['Close'][0]) / r_actual['Close'][0] * 100.))
print("Predictions: ", predictions)

print("--------------------------------------------")

# get a row number
n_actual = n_stock.loc[n_stock.index == '03-29-2017']
row = n_stock.index.get_loc('03-29-2017')

# Run data through prediction models
predict_w = n_week.predict((n_stock.ix[row + one_week])[features])
predict_m = n_month.predict((n_stock.ix[row + one_month])[features])
predict_q = n_quarter.predict((n_stock.ix[row + one_quarter])[features])

predictions = [predict_w[0], predict_m[0], predict_q[0]]

print("NASDAQ Actual Mar 29th, 2017: ", n_actual['Close'])
print("Std: ", np.std(predictions))
print("Mean: ", np.mean(predictions))
print("Diff: %.2lf%%" % (np.mean(predictions - n_actual['Close'][0]) / n_actual['Close'][0] * 100.))
print("Predictions: ", predictions)
