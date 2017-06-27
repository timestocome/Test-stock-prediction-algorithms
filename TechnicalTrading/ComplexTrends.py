# https://github.com/timestocome


# tried looking at all sorts of known patterns in stocks
# I couldn't find the patterns people claim to find
# here you can create and look for patterns yourself

# try a more complicated pattern search
# instead of just gain or loss, look
# for high and low gain or loss patterns in the 
# time series


import pandas as pd
import numpy as np
from collections import Counter
from collections import OrderedDict
import operator



#########################################################################
# user vars
########################################################################
high_gain = 0.05        # returns above this count as high return days
high_loss = -0.05       # returns below this count as high loss days
window = 10             # number of days to look for pattern over



#########################################################################
# utility functions
########################################################################

def find_the_key(dictionary, thekey):

    for k, v in dictionary.items():

        p1 = list(k)[:-1]
        p2 = list(thekey)[:-1]

        if k == thekey: 
            return v

##########################################################################
# read in data (daily price of the stock)
##########################################################################

# read in NASDAQ
data = pd.read_csv('data/nasdaq.csv', parse_dates=True, index_col=0)
data = data[['High', 'Low', 'Open']]
data['Open'] = pd.to_numeric(data['Open'], errors='coerce')
data['LogOpen'] = np.log(data['Open'])


data['dx'] = data['LogOpen'] - data['LogOpen'].shift(1)
data = data.dropna()



# check count below and adjust if needed 
def dailyChange(d):
    
    if d >= high_gain: return 2      # high return
    elif d >= 0.0: return 1
    elif d <= high_loss: return -2  # high loss
    else: return -1

data['change'] = data['dx'].apply(dailyChange)


print("Number of high, low gain and loss days:")
print(data['change'].value_counts())
print("    ")



######################################################################
# first attempt at finding trends
window = window + 1         # try a x day sliding window + plus one to see
                    # next day gain or loss
######################################################################


# look for patterns in the gains, losses
# slide window over data and collect patterns
collect_patterns = []
pattern = []
changes = data['change'].values

n_changes = len(changes)

for i in range(len(changes) - window):
    collect_patterns.append(changes[i:i+window])
    
    
# pull out unique patterns
unique_patterns = set(tuple(z) for z in collect_patterns)


# get frequency counts
pattern_frequency = Counter(tuple(z) for z in collect_patterns)

# need to find both halves of matching patterns, 
# window -1  days of pattern
# last digit is the result gain or loss
sorted_list = sorted(pattern_frequency.items(), key=operator.itemgetter(0))

print("Unique patterns found: %d of %d possible patterns: " % (len(sorted_list), 4 ** window) )
print("   ")

print("Searching for patterns %d days long: " % (window-1))
print("High gain: 2, small gain: 1, small loss: -1, large loss: -2")
print("-------------------------------------------------------------")


low_gains = []
high_gains = []
low_losses = []
high_losses = []

# split out patterns
for k, v in sorted_list:
    
    pattern = list(k)[:-1]
    future = list(k)[-1]
    
    if future == 2: high_gains.append(k)
    if future == -2: high_losses.append(k)
    if future == 1: low_gains.append(k)
    if future == -1: low_losses.append(k)


print("--------------------------------------------------------------------------------------------------")
print("Patterns resulting in large gains")
print("--------------------------------------------------------------------------------------------------")
# get frequency counts
high_return_patterns_frequency = Counter(tuple(z) for z in high_gains)
sorted_high_return_patterns = sorted(high_return_patterns_frequency.items(), key=operator.itemgetter(0))

for k, v in sorted_high_return_patterns:
    total_times = find_the_key(pattern_frequency, k)
    p = list(k)[:-1]
    print("Number of times pattern appears before high return day: %d, total times: %d %s" % (v, total_times, p))

print("--------------------------------------------------------------------------------------------------")
print("Patterns resulting in large losses")
print("--------------------------------------------------------------------------------------------------")

# get frequency counts
high_loss_patterns_frequency = Counter(tuple(z) for z in high_losses)
sorted_high_loss_patterns = sorted(high_loss_patterns_frequency.items(), key=operator.itemgetter(0))

for k, v in sorted_high_loss_patterns:
    total_times = find_the_key(pattern_frequency, k)
    p = list(k)[:-1]
    print("Number of times pattern appears before high loss day: %d, total times: %d %s" % (v, total_times, p))


print("--------------------------------------------------------------------------------------------------")
print("All other patterns")
print("--------------------------------------------------------------------------------------------------")


p_previous = []
results = []
frequencies = []
x = ""
for k, v in sorted_list:

    p = k[:-1]
    r = k[-1]
    #print(p, r, v)

    if len(p_previous) > 0:
        
        if p != p_previous:

            # patterns that only occur once aren't useful:
            if len(frequencies) > 1:
                print('*********************************************************')
                print(p)
                for i in range(len(frequencies)):
            
                    if results[i] == 2: x = 'High gain'
                    elif results[i] == 1: x = 'Low gain'
                    elif results[i] == -1: x = 'Small loss'
                    elif results[i] == -2: x = 'Large loss'
                    print("Occurs: %d times resulting in %s " % (frequencies[i], x))

            results = [r]
            frequencies = [v]
        else:
           results.append(r)
           frequencies.append(v)
    
    
    p_previous = p    

