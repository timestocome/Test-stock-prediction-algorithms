# https://timestocome/github.com



# simple markov chain 


import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt 



# set up 
rng = np.random.RandomState(27)

###############################################################################################
# read in and set up data
###############################################################################################

data = pd.read_csv('StockDataWithVolume.csv', parse_dates=True, index_col=0)



# convert to log scale and calculate daily difference in value
log_dx_data = data
columns = log_dx_data.columns
for col in columns:
    log_dx_data[col] = pd.to_numeric(log_dx_data[col], errors='coerce')
    log_dx_data[col] = np.log(log_dx_data[col])
    log_dx_data[col] = log_dx_data[col] - log_dx_data[col].shift(1)
    log_dx_data[col] = log_dx_data[col].dropna()





#print(merged_data)

# pick an index and a likely indicator
# print( (merged_data.corr()['Gold']))  # djia, 1 yr tbills strongest correlations with gold
Gold = log_dx_data[['Gold']]


# What we'd like to predict
states = ('up_today', 'down_today')


# what we know
observations = ('up_yesterday', 'down_yesterday')


# calculate probabilities
up = 0
down = 0
total = len(Gold)

for ix, row in Gold.iterrows():
    if row.Gold > 0: up += 1
    else: down += 1
up /= total 
down /= total

overall_probabilities = { 'up_today': up, 'down_today': down }

print("Probability up %.3f, down %.3f" % (up, down))

# transition probability
down_down = 0
down_up = 0
up_down = 0
up_up = 0
previous = Gold.iloc[0].Gold
current = Gold.iloc[0].Gold 

for ix, row in Gold.iterrows():

    current = row.Gold 
    if current <= 0:
        if previous <= 0:    down_down += 1
        elif previous > 0:  up_down += 1

    if current > 0:
        if previous <= 0:    down_up += 1
        elif previous > 0:  up_up += 1

    previous = current

down_down /= (down * total)
down_up /= (down * total)
up_down /= (up * total)
up_up /= ( up * total)

transition_probabilities = {
    'up_yesterday': {'up_today': up_up, 'down_today': up_down},
    'down_yesterday': {'up_today': down_up, 'down_today': down_down }
}

# up == 0
# down == 1
probabilities = np.asarray( [[up_up, up_down], 
                            [down_up, down_down]])

print("   ")
print('Probabilities of transitions:')
print(probabilities)
# yesterday
x_up = [0, 1]
x_down = [1, 0]


# tomorrow predictions
print("     ")
print("Tomorrow's probabilities:")
print("If today is up tomorrow up %.3lf, tomorrow down %.3lf" %(probabilities[0,0], probabilities[0,1]))
print("If today is down tomorrow up %.3lf, tomorrow down %.3lf" %(probabilities[1,0], probabilities[1,1]))

# day after tomorrow predictions
print("     ")
print("Two days from now:")
probabilities2 = probabilities **2
print("If today is up in 2 days up %.3lf, in 2 days down %.3lf" % 
            (probabilities2[0,0] / (probabilities2[0,0] + probabilities2[0,1]), +
            probabilities2[0,1] / (probabilities2[0,0] + probabilities2[0,1]))) 

print("If today is down in 2 days up %.3lf, in 2 days down %.3lf" % 
            (probabilities2[1,0] / (probabilities2[1,0] + probabilities2[1,1]), 
             probabilities2[1,1] / (probabilities2[1,0] + probabilities2[1,1])))




