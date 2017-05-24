# http://github.com/timestocome


# use hidden markov model to predict changes in a stock market index fund
# http://cs229.stanford.edu/proj2009/ShinLee.pdf
# https://www.dartmouth.edu/~chance/teaching_aids/books_articles/probability_book/Chapter11.pdf


import numpy as np
import pandas as pd
import tensorflow as tf
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

'''
############################################################################
# plot returns on NASDAQ training data
#############################################################################

fig = plt.figure(figsize=(10,10))
plt.subplot(2,1,2)


plt.subplot(2,1,1)
plt.plot(data['returns'])
plt.title("Nasdaq daily returns")


# histogram of returns
plt.subplot(2,1,2)
plt.hist(data['returns'], bins=200)
plt.xlabel("Returns")
plt.ylabel("Probability")
plt.title("Histogram daily Nasdaq returns")
plt.grid(True)

# median
median_return = data['returns'].median()
l = plt.axvspan(median_return-0.0001, median_return+0.0001, color='red')

plt.show()
'''
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


###################################################################
# Simple markov chain
###################################################################
# probabilities of high/low gains, high/low losses per day
states = ['HighGain', 'LowGain', 'LowLoss', 'HighLoss']

# divide gains into highGain, lowGain, lowLoss, highLoss
def sectionReturns(r):
    if r > 0.05: return 4
    elif r > 0.00: return 3
    elif r > -0.05: return 2
    else: return 1

data['gainLoss'] = data['returns'].apply(sectionReturns)
total_samples = len(data)

counts = data.groupby(['gainLoss']).count()
counts = counts['NASDAQ']

# probability of each return on a random day
probability = np.array([0., 0., 0., 0.])
probability[0] = counts[1] / total_samples        # dataframe, not array start with 1
probability[1] = counts[2] / total_samples
probability[2] = counts[3] / total_samples
probability[3] = counts[4] / total_samples


# probabilities of changing from one state to the next tomorrow
# transition maxtrix going from previous day's state to next on a given day
# hg-hg, hg-lg, hg-ll, hg-hl
# lg-hg, lg-lg, lg-ll, lg-hl
# ll-hg, ll-lg, ll-ll, ll-hl
# hl-hg, hl-lg, hl-ll, hl-hl
transitions = np.zeros((4, 4))

def transition(current_return, previous_return):

    if current_return > 0.05: 
        if previous_return > 0.05:      transitions[0][0] += 1
        elif previous_return > 0.00:    transitions[0][1] += 1
        elif previous_return > -0.05:   transitions[0][2] += 1
        else:                           transitions[0][3] += 1

    elif current_return > 0.00:         
        if previous_return > 0.05:      transitions[1][0] += 1
        elif previous_return > 0.00:    transitions[1][1] += 1
        elif previous_return > -0.05:   transitions[1][2] += 1
        else:                           transitions[1][3] += 1
    elif current_return > -0.05: 
        if previous_return > 0.05:      transitions[2][0] += 1
        elif previous_return > 0.00:    transitions[2][1] += 1
        elif previous_return > -0.05:   transitions[2][2] += 1
        else:                           transitions[2][3] += 1
    else: 
        if previous_return > 0.05:      transitions[3][0] += 1
        elif previous_return > 0.00:    transitions[3][1] += 1
        elif previous_return > -0.05:   transitions[3][2] += 1
        else:                           transitions[3][3] += 1


total_transitions = 0
for i in range(len(data)-1):
    total_transitions += 1
    previous = data.iloc[i]['returns']
    current = data.iloc[i+1]['returns']
    transition(current, previous)

# normalize matrix, then normalize rows
transitions /= total_transitions
transitions /= transitions.sum(axis=0)    

#######################################################################################
# make a prediction
# n number of days into future
# s today's state hg, lg, ll, hl
# t transition matrix that was calculated
s = 1
n = 5
t = transitions
prediction_probability = (t **n)


print("%d days from now, the market return will be %s" % (n, states[prediction_probability[s].argmax()]))


######################################################################################
# plot predictions over time

# scale prediction for plotting
def convert_return_for_plot(r):
    if r == 0: return 0.1 * 0.1
    if r == 1: return 0.5 * 0.1
    if r == 2: return -0.5 * 0.1
    if r == 3: return -1. * 0.1


days_ahead = 5
p = []

for i in range(len(data)-1):
    s = int(data.iloc[i]['gainLoss']) - 1
    prediction_probability = (transitions **days_ahead)        # predict all states at future data
    prediction = prediction_probability[s].argmax()            # get states with today's starting condition
    scaled_return = convert_return_for_plot(prediction)
    p.append(scaled_return)                                       # 1 hg, 2, lg, 3, ll, 4 hl      

# pad begining of p 
p = ([0] * 1 + p)
data['predicted'] = p

plt.figure(figsize=(12,12))
plt.title("Nasdaq daily gain/loss using single chain markov 5 days out")
plt.plot(data['returns'], label='Actual')
plt.plot(data['predicted'], label='Predicted', alpha=0.5)
plt.legend(loc='best')
plt.savefig("SingleChainMarkov.png")
plt.show()