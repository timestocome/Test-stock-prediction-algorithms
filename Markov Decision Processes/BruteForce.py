# http://github.com/timestocome

# adapted from:
#    Planning with Markov Decision Processes, An AI Perspective

# This is the first and simplest example from section 3.1
# Hoping for a smarter MDP as the algorithms get more advanced
#
# randomly try buy, sell, hold depending on state, 
# keep score of what returns the highest reward in each state


from Data import Data
from Bot import Bot

import numpy as np
import pandas as pd


###############################################################################
# actions
# buy with all of cash on hand, hold, sell all shares on hand
###############################################################################
actions = np.array(['Buy', 'Hold', 'Sell'])
n_actions = len(actions)
buy = 0
hold = 1
sell = 2

###############################################################################
# data stream
# decide how many states to use, the more states used, the more data is needed
###############################################################################
states = np.array(['high_loss', 'low_loss', 'no_change', 'low_gain', 'high_gain'] )
n_states = len(states)

signal = Data('btc.csv')
data = signal.data

# bucket daily change into states
binned_data = pd.cut(data['dx'], n_states, labels=states, retbins=True)
dx = binned_data[0]
edges = binned_data[1]
n_samples = len(dx)

# print(dx)
# print(edges)


# create matrix containing transition probabilities
# rows are from state, columns are to state
transitions = np.zeros((n_states, n_states))
for i in range(0, n_samples-1):
    t_from = np.where(states == dx[i])[0][0]
    t_to = np.where(states == dx[i+1])[0][0]
    transitions[t_from][t_to] += 1
transition_probability = transitions / n_samples  
    
# print(transitions)



###############################################################################
# Policy V_Pi
# Possiblities S^A ~ 5 states ^ 3 actions ~ 125 policies
###############################################################################

# current state -> next state = reward
policy = np.zeros((n_states, n_actions))


###############################################################################
# run through data for n_training runs
# randomly select actions and save profit/loss info as rewards
###############################################################################

bot = Bot()
v_pi = 0

n_training_runs = 200

for t in range(n_training_runs):
    print('.... training ', t)

    for i in range(n_samples - 1):

        s1 = np.where(states == dx[i])[0][0]        # current state
        s2 = np.where(states == dx[i+1])[0][0]      # next state
        a = np.random.randint(n_actions)            # action to take
        p = data['Price'].iloc[i]                   # current price
        v_pi += 1                                   # time step cost
    
        reward = bot.move(p, a, s1)                 # reward when sell coins
        if reward != 0: v_pi = 0                    # reset time step 
    
    
        policy[s1][a] += transition_probability[s1][s2] * (reward + v_pi)    
    



###############################################################################
# save policy
###############################################################################

np.save('brute_force.npy', policy)

print('----------------------------------------------------------------------')
print('Saved policy: rows are states, columns are actions' )
print('Select highest action in row for state')
print(policy)



###############################################################################
# test policy
###############################################################################
print('----------------------------------------------------------------------')
print('Test saved policy at random starting locations')
print('Set starting cash, and run length for number of days to test policy')


saved_policy = np.load('brute_force.npy')


n_test_runs = 20       # how many times to test policy
starting_cash = 100.   # beginning cash
run_length = 21        # how many days to run test


print('Starting cash $ %.2f' % starting_cash)
print('Test for %d days' % run_length)


print('----------------------------------------------------------------------')


for t in range(n_test_runs):

    cash_on_hand = starting_cash
    shares_on_hand = 0.

    # pick a random start location
    idx = np.random.randint(n_samples - run_length)

    buys = 0
    sells = 0

    for i in range(idx, idx + run_length):

        s1 = np.where(states == dx[i])[0][0]
        a = np.argmax(saved_policy[s1][:])

        if a == 0:          # buy
            if cash_on_hand > 0:
                shares_on_hand = cash_on_hand / data['Price'].iloc[i]
                cash_on_hand = 0.
                buys += 1
            
            elif a == 2:        # sell
                if shares_on_hand > 0:
                    cash_on_hand = shares_on_hand * data['Price'].iloc[i]
                    shares_on_hand = 0.
                    sells += 1


    # cash out
    profits = shares_on_hand * data['Price'].iloc[idx + run_length] + cash_on_hand
    print('Ending cash $ %.2f, Buys %d Sells %d' % (profits, buys, sells))



