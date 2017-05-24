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


###################################################################
# Simple markov chain
###################################################################

# first pass only used 4 bins ( highGain, lowGain, lowLoss, highLoss )
# looks to be ~0.13, -.112, ~.25 diff between highest and lowest

# divide returns into bins
# round(2) gives 22 unique bins
# round(3) gives 157 bins
# round(4) gives 848 bins
round_values = 4
data['gainLoss'] = data['returns'].round(round_values)

total_samples = len(data)
n_bins = data['gainLoss'].nunique()
value_count = data['gainLoss'].value_counts()
value_count = value_count.sort_index()
b = value_count.index.tolist()
bins = ['%.4f' % z for z in b]          # match to round value


#print(value_count)

# calculate probability of a return value on a random day
probability = value_count / total_samples
#print(probability)


# built transition matrix
transitions = np.zeros((n_bins, n_bins))
def map_transition(this_return, previous_return):
    current =  np.where(probability.index==this_return)[0] - 1      # pandas starts at 1, numpy starts at zero
    previous = np.where(probability.index==previous_return)[0] - 1
    transitions[current, previous] += 1


total_transitions = 0
for i in range(len(data)-1):
    total_transitions += 1
    previous = data.iloc[i]['gainLoss']
    current = data.iloc[i+1]['gainLoss']
    map_transition(current, previous)

# normalize matrix, then normalize rows
transitions /= total_transitions
transitions /= transitions.sum(axis=0)    

#######################################################################################
# make a prediction
# n number of days into future
# s today's state hg, lg, ll, hl
# t transition matrix that was calculated
s = -.03     # today's gain or loss --- be sure it is a valid bin
n = 5
t = transitions
prediction_probabilities = (t **n)
row_number = np.where(probability.index==s)[0] - 1      # pandas starts at 1, numpy starts at zero
probabilities = prediction_probabilities[row_number]
mostlikely = probabilities.argmax()
bin_value = float(bins[mostlikely]) 

print("%d days from now, the market return will be %.2f" % (n, bin_value))


######################################################################################
# plot predictions over time

# scale prediction for plotting
def convert_return_for_plot(r):
    return bins[r]

days_ahead = 5
p = []


for i in range(len(data)-1):
    s = data.iloc[i]['gainLoss']                        # get current day return from market
    prediction_probabilities = (transitions **n)                          # predict all probabilities for future date
    row_number = np.where(probability.index==s)[0] - 1          # get row number matching today's return
    probabilities = prediction_probabilities[row_number]
    mostlikely = probabilities.argmax()
    bin_value = bins[mostlikely]
    p.append(bin_value)


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

