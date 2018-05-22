
# http://github.com/timestocome

# adapted from:
#  https://github.com/maxpumperla/betago
#  https://www.manning.com/books/deep-learning-and-the-game-of-go







class Bot(object):
    
    
    def __init__(self, starting_cash=1000., starting_shares=0., n_states=5, n_actions=3):
        
        self.cash = starting_cash
        self.shares = starting_shares
        self.purchase_price = 0.
        
        
    def move(self, price, action, state):    
        
        reward = 0.
        
        # buy
        if action == 0:     
            if self.cash > 0.:
                self.shares += self.cash / price
                self.cash = 0.
                self.purchase_price = price * self.shares
            
        # hold
        elif action == 1:
            pass
            
        
        # sell    
        elif action == 2:
            if self.shares > 0.:
                self.cash += self.shares * price
                self.shares = 0.    
        
                reward = (self.purchase_price - self.cash) / self.purchase_price
    
    
        return reward
        
    
   