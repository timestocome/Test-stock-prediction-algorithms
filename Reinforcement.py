
# http://github.com/timestocome



# started with this code from book
# fixed bugs, everything is working
# streamlined code
# added plots
# improved commenting

# https://github.com/BinRoot/TensorFlow-Book/blob/master/ch08_rl/rl.py


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import random



##############################################################################
# load data
#############################################################################
# data is just a list of opening prices
# nasdaq.csv was downloaded from finance.yahoo.com
def load_prices():

    data_in = pd.read_csv('nasdaq.csv', header=0)
    data_in = data_in[['Open']]

    data = np.asmatrix(data_in)
    n_test = len(data) // 10
    n_train = len(data) - n_test
    train = data[0:n_train]
    test = data[n_train:-1]

    # have to do some array mangling in training loop, need to start with lists
    return train.tolist(), test.tolist()



def plt_prices(prices):
    plt.title('Opening stock prices')
    plt.xlabel('days')
    plt.ylabel('price')
    plt.plot(prices)
    plt.savefig('prices.png')
    plt.show()



train, test = load_prices()
#plt_prices(train)

##############################################################################
# network
#############################################################################

# randomly choose an action ( buy, sell, hold )
class RandomDecisionPolicy():
    
    def __init__(self, actions):
        self.actions = actions

    
    def select_action(self, current_state, step):
        action = self.actions[random.randint(0, len(self.actions) - 1)]
        return action

    
    def update_q(self, state, action, reward, next_state):
        pass




class QLearningDecisionPolicy():

    def __init__(self, actions, n_input):
        
        self.epsilon = 0.9      # how frequently to try a random action 1-epsilon == random %   
        self.gamma = 0.001      # how far back to remember
        self.actions = actions
        
        n_output = len(actions)
        n_hidden = n_input - 2      # budget and n_stocks are tacked onto end of input


        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.y = tf.placeholder(tf.float32, [n_output])

        
        W1 = tf.Variable(tf.random_normal([n_input, n_hidden]))
        b1 = tf.Variable(tf.constant(0.1, shape=[n_hidden]))
        h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)

        W2 = tf.Variable(tf.random_normal([n_hidden, n_output]))
        b2 = tf.Variable(tf.constant(0.1, shape=[n_output]))

        self.q = tf.nn.relu(tf.matmul(h1, W2) + b2)

        loss = tf.square(self.y - self.q)
            
        self.train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())



    def select_action(self, current_state, step):

        threshold = min(self.epsilon, step / 1000.)
        
        # if random number (0-1) > epsilon .9 try a random move ~10%
        if random.random() < threshold:     # take best known action

            action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
            action_idx = np.argmax(action_q_vals)  
            action = self.actions[action_idx]
        
        else:                               # random
            action = self.actions[random.randint(0, len(self.actions) - 1)]
        return action




    def update_q(self, state, action, reward, next_state):

        action_q_vals = self.sess.run(self.q, feed_dict={self.x: state})
        next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state})

        next_action_idx = np.argmax(next_action_q_vals)
        action_q_vals[0, next_action_idx] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
        action_q_vals = np.squeeze(np.asarray(action_q_vals))
        
        self.sess.run(self.train_op, feed_dict={self.x: state, self.y: action_q_vals})






def run_simulation(policy, budget, n_stocks, prices, history):

    
    share_value = 0
    transitions = list()
    n_simulations = len(prices) - history - 1
    actions_taken = []      # save the last run of actions for plotting
    

    for i in range(n_simulations):


        # painful but necessary array shape manipulations
        budget = np.array([budget]).reshape(1,1)
        n_stocks = np.array([n_stocks]).reshape(1,1)
        i_prices = np.array(prices[i+1 : i+1+history]).T
        
        current_state = np.asmatrix(np.hstack((i_prices, budget, n_stocks)))

        
        current_portfolio = budget + n_stocks * share_value
        action = policy.select_action(current_state, i)
        share_value = prices[i + history + 1]
    
        actions_taken.append(action)

        if action == 'Buy' and budget >= share_value:
            budget -= share_value
            n_stocks += 1

        elif action == 'Sell' and n_stocks > 0:
            budget += share_value
            n_stocks -= 1

        else:
            action = 'Hold'
        
        new_portfolio = budget + n_stocks * share_value
        reward = new_portfolio - current_portfolio

    
        # painful but necessary array shape manipulations
        next_state = i_prices
        budget = np.array([budget]).reshape(1,1)
        n_stocks = np.array([n_stocks]).reshape(1,1)
        next_state = np.asmatrix(np.hstack((i_prices, budget, n_stocks)))


        transitions.append((current_state, action, reward, next_state))
        policy.update_q(current_state, action, reward, next_state)

    portfolio = budget + n_stocks * share_value
    
    return portfolio, actions_taken



def run_simulations(policy, budget, n_stocks, prices, history):

    n_simulations = 10
    final_portfolios = list()
    final_actions = list()
    
    for i in range(n_simulations):

        final_portfolio, final_actions = run_simulation(policy, budget, n_stocks, prices, history)
        final_portfolios.append(final_portfolio)
        final_actions.append(final_actions)

    return np.mean(final_portfolios), np.std(final_portfolios), final_actions






################################################################################
# run simulations
################################################################################

print("*********   run simulations ***********")


prices, test = load_prices()

actions = ['Buy', 'Sell', 'Hold']
history = 21        # how large of a window of stock prices to view ( 21/mth, 63/qtr, 251/yr )
budget = 1000.0     # begining cash on hand
n_stocks = 0        # begining shares on hand



# buy on day one and hold
buy_and_hold = budget * (prices[-1][0] - prices[0][0])
print("Buy all and hold on day one Init: $1000, Profit: %.2f" % buy_and_hold)


# try random first
policy = RandomDecisionPolicy(actions)
avg, std, _ = run_simulations(policy, budget, n_stocks, prices, history)
print("Random trades: Init: $1000, Avg profit: $%.2f, Std: $%.2f " %(avg, std))
    

# reset init conditions and try RL learning    
budget = 1000.0
n_stocks = 0
    
policy = QLearningDecisionPolicy(actions, history + 2)
avg, std, actions = run_simulations(policy, budget, n_stocks, prices, history)
print("Q Learning trades: Init: $1000, Avg profit: $%.2f, Std: $%.2f " %(avg, std))



####  plot last run actions ##########

# trading doesn't begin till we hit our history size
buys = np.zeros(len(prices))
sells = np.zeros(len(prices))

for i in range(history, len(prices)-1):
    if actions[i-history] == 'Buy': buys[i] = 1 * prices[i][0]
    if actions[i-history] == 'Sell': sells[i] = 1 * prices[i][0]

x = np.arange(0, len(prices))

plt.figure(figsize=(24,16))
plt.title('RL Bot stock trades')
plt.xlabel('days')
plt.ylabel('price')
#plt.plot(prices)
# it's not easy to see buys and sell on entire graph at once, better to use windows
plt.scatter(x[0:1000], buys[0:1000], c='green', alpha=1., s=8.)
plt.scatter(x[0:1000], sells[0:1000], c='red', alpha=1., s=8.)
plt.savefig('bot_trades.png')
plt.show()