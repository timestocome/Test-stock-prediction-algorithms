# http://github.com/timestocome

# adapted from:
#    Planning with Markov Decision Processes, An AI Perspective

# This is the second example from section 3.4
#  Adds in recursive Bellman equation
#
# Value iteration
#
# V*(s) = 0
# Q*(s,a) = Sum(transition_probability * (reward + V*(s')))
#
# Vn(s) = max(sum(transition_probability(s, s') * reward(s,a,s') + V*previous(s')))





from Data import Data

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
# Policy 
# 
###############################################################################

# current state -> next state = reward
policy = np.random.random((n_states, n_actions))
value = np.zeros(n_states)



# init

# measure progress
epsilon = 0.01       # changes in policy
diff = 9999.999      # changes in policy
previous_policy = np.copy(policy)

# starting state
cash_on_hand = 1000.
coins_on_hand = 10.


for z in range(100):
#while diff > epsilon:
    
    previous_policy = np.copy(policy)


    # run through all Coin/Stock series in sequencial order
    for n in range(n_samples-1):
        
        #for s in range(n_states):
            policy_before = np.copy(policy)
            
            # get current and next state ( change in daily price )
            s_now = np.where(states == dx[n])[0][0]
            s_next = np.where(states == dx[n])[0][0]
            
            # current state
            #for a in range(n_actions):
            # calculate relative value of coins and cash each day
            b1_coins = data['Price'].iloc[n] / cash_on_hand
            b1_cash = cash_on_hand
            
            h1_coins = coins_on_hand
            h1_cash = cash_on_hand
            
            s1_coins = coins_on_hand
            s1_cash = coins_on_hand * data['Price'].iloc[n]
            
            # next state
            b2_coins = data['dx'].iloc[n+1] / cash_on_hand
            b2_cash = cash_on_hand
            
            h2_coins = coins_on_hand
            h2_cash = cash_on_hand
            
            s2_coins = coins_on_hand
            s2_cash = coins_on_hand * data['Price'].iloc[n+1]
            
            # rewards ( normalize sell data ) * transition probability of s_now -> s_next
            reward_buy = ((b2_coins - b1_coins) * data['dx'].iloc[n+1] + (b2_cash - b1_cash)) * transition_probability[s_now][s_next]
            reward_hold = ((h2_coins - h1_coins) * data['dx'].iloc[n+1] + (h2_cash - h1_cash)) * transition_probability[s_now][s_next]
            reward_sell = (((s2_coins - s1_coins) * data['dx'].iloc[n+1] + (s2_cash - s1_cash)) / cash_on_hand ) * transition_probability[s_now][s_next]
            
            value[s_now] = max(reward_buy, reward_hold, reward_sell)
            
            
            # add in value of being in state s_next
            reward_buy += value[s_next]
            reward_hold += value[s_next]
            reward_sell += value[s_next]
            
            
                                    
            # update policy if things worked out better
            if policy[s_now][0] < reward_buy: policy[s_now][0] = reward_buy
            if policy[s_now][1] < reward_hold: policy[s_now][1] = reward_hold
            if policy[s_now][2] < reward_sell: policy[s_now][2] = reward_sell
            
         
    # check progress
    diff = np.sum(np.abs(policy - previous_policy))
    print('.... training ', z)

    


###############################################################################
# save policy
###############################################################################

np.save('value_iteration.npy', policy)

print('----------------------------------------------------------------------')
print('Saved policy: rows are states', states) 
print('columns are actions', actions )
print('Select highest action in row for state')


print(policy)
for i in range(n_states):
    best_a = np.argmax(policy[i][:])
    if best_a == 0:
        print(states[i], 'buy')
    if best_a == 1:
        print(states[i], 'hold')
    if best_a == 2:
        print(states[i], 'sell')




###############################################################################
# test policy
###############################################################################
print('----------------------------------------------------------------------')
print('Test saved policy at random starting locations')
print('Set starting cash, and run length for number of days to test policy')


saved_policy = np.load('value_iteration.npy')


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



