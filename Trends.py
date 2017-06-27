# https://github.com/timestocome


# tried looking at all sorts of known patterns in stocks
# I couldn't find the patterns people claim to find
# here you can create and look for patterns yourself



import pandas as pd
import numpy as np
from collections import Counter
from collections import OrderedDict
import operator





##########################################################################
# read in data (daily price of the stock)
##########################################################################

# read in NASDAQ
data = pd.read_csv('data/nasdaq.csv', parse_dates=True, index_col=0)
data = data[['High', 'Low', 'Open']]
data['Open'] = pd.to_numeric(data['Open'], errors='coerce')


data['dx'] = data['Open'] - data['Open'].shift(1)
data = data.dropna()


######################################################################
# first attempt at finding trends
window = 6          # try a x day sliding window
######################################################################
# convert daily change to -1, 1 to simplify things
data['change'] = np.where(data['dx'] > 0, 1, -1)

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

print("Unique patterns found %d from %d possible patterns: " % (len(sorted_list), 2 ** window) )


print("Searching for patterns %d days long: " % window)
print("Daily gain +1, loss -1")



last_key = []
last_value = 0
for k, v in sorted_list:
    
    if last_value > 0:

        p1 = list(k)[:-1]
        p2 = list(last_key)[:-1]    # get pattern leading up to gain/loss

        gain = v                    # get the gain or loss, data is sorted descending
        loss = last_value           # ... so gain(1) aways first in match
        frequency = gain + loss

        print("    ")

        if p1 == p2:
            print('Pattern appears: %.2lf%% of the time ' % (frequency/n_changes * 100.))

            if gain > loss:
                print("Pattern: %s Next day gain occurs: %.3lf%% of the time" % (p1, gain/(loss + gain) * 100.) )
            else:
                print("Pattern: %s Next day loss occurs: %.3lf%% of the time" % (p1, loss/(loss + gain) * 100.) )
        else:
             if frequency > 2: # loop counts unique patterns twice in frequency
                print('Pattern appears once')

                if gain > loss:
                    print("Pattern: %s Next day gain occurs: 100%% of the time" % p1)
                else:
                    print("Pattern: %s Next day loss occurs: 100%% of the time" % p1)




    last_key = k 
    last_value = v

