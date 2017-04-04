
# http://github.com



# Working through MEAP Machine Learning w/ TensorFlow Book
# added a few things to their sample code 

# first pass stock estimates using random policy reinforcement learning 



import tensorflow as tf 
import numpy as np
import random 

from yahoo_finance import Share 

import matplotlib.pyplot as plt




def get_prices (share_symbol, start_date, end_date, cache_filename='stock_prices.npy'):

    try:
        stock_prices = np.load(cache_filename)

    except IOError:
        share = Share(share_symbol)
        stock_hist = share.get_historical(start_date, end_date)
        stock_prices = [stock_price['Open'] for stock_price in stock_hist]
        np.save(cache_filename, stock_prices)

    return stock_prices


def plot_prices(prices):

    plt.title('Opening stock prices')
    plt.xlabel('Day')
    plt.ylabel('Price $')
    plot.plot(prices)
    plt.savefig('prices.png')



class DecisionPolicy:

    def select_action(self, current_state):
        pass

    def update_Q(self, state, action, reward, next_state):
        pass 




class RandomDecisionPolicy(DecisionPolicy):

    def __init__(self, actions):
        self.actions = actions

    def select_action(self, current_state, step):
        action = self.actions[random.randint(0, len(self.actions) - 1)]
        return action 





def run_simulation(policy, initial_budget, initial_number_stocks, prices, history, debug=False):

    budget = initial_budget
    n_stocks = initial_number_stocks
    share_value = 0
    plan = []


    transitions = list()
    for i in range(len(prices) - history - 1):

        current_state = np.asmatrix(np.hstack((prices[i:i+history], budget, n_stocks)))
        current_portfolio = budget + n_stocks + share_value
        action = policy.select_action(current_state, i)
        share_value = float(prices[i + history + 1])

        if action == 'Buy' and budget >= share_value:
            budget -= share_value 
            n_stocks += 1

        elif action == 'Sell' and n_stocks > 0:
            budget += share_value
            n_stocks -= 1 

        else:
            action == 'Hold'

        new_portfolio = budget + n_stocks * share_value
        reward = new_portfolio - current_portfolio
        next_state = np.asmatrix(np.hstack((prices[i+1:i+history+1], budget, n_stocks)))
        transitions.append((current_state, action, reward, next_state))

        policy.update_Q(current_state, action, reward, next_state)
        plan.append((action, share_value))
        
    portfolio = budget + n_stocks * share_value
    return portfolio, plan



def run_simulations(policy, budget, n_stocks, prices, history):
    n_tries = 10
    final_portfolios = list()
    final_policies = list()
    final_plans = list()

    for i in range(n_tries):
        final_portfolio, final_plan = run_simulation(policy, budget, n_stocks, prices, history)
        final_portfolios.append(final_portfolio)
        final_plans.append(final_plan)

    avg, std = np.mean(final_portfolios), np.std(final_portfolios)
    return avg, std, final_plans






##################################################################################################

actions = ['Buy', 'Sell', 'Hold']
policy = RandomDecisionPolicy(actions)

budget = 1000.
n_stocks = 0
prices = get_prices('MSFT', '1992-01-01', '2017-04-01')
history = 200

avg, std, plans = run_simulations(policy, budget, n_stocks, prices, history)


last_plan = plans[len(plans) -1]
prices = prices[history:-1]

fig = plt.figure(figsize=(24,16))
ax = fig.add_subplot(1,1,1)
d = 0
for p in last_plan:
    action, price = p 
    d += 1
   
    if action == 'Buy': 
        ax.scatter(d, price, c='green', alpha=0.7, s=12)
    if action == 'Sell': 
        ax.scatter(d, price, c='red', alpha=0.7, s=12)


years = len(last_plan) / 251
earnings = (avg - budget) / budget 
returns = np.power((1 + earnings), (1./years))

print("Avg earnings", avg)
print("Std", std)
print("Years invested", years)
print("Total earnings %.2lf%%" % (earnings) )
print("Average earnings ~ %.2lf%%" %(returns) )
plt.title("Random policy")
subtitle = 'Buys are green, Sells are red, avg return on $1000   = %d over %d trading days' % (int(avg), len(last_plan))
plt.suptitle(subtitle)
plt.plot(prices, c='black')
plt.show()

