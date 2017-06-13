
# http://github.com/timestocome


# convert Fully Connected neural network used in the sunspot predicitions
# to work with stock index data

# Fully connected network written from sunspot paper
# https://github.com/timestocome/Sunspots

# data files and cleanup code (LevelData.py)
# https://github.com/timestocome/StockMarketData

# sunspot paper
# http://surface.syr.edu/cgi/viewcontent.cgi?article=1056&context=eecs_techreports






# Todo:
# paper uses LR 0.3, momentum 0.6, epochs 5000-30,000
# regularization ? drop out, L1? L2
# second prediction loop that uses the previous prediction, not known value to predict next value
# so we can see how far it wanders for predictions longer than one look ahead.
#
#
# Using the data from the peak finding algorithm
# stock data has been rotated to x-axis using linear regression 
# log values of stock are used
# convert data back from rotation around x axis
# convert data back from log values
# remove +1 added to bring data from -1 to 1 above zero
# data['leveled log Nasdaq'] = data['log NASDAQ'] - (6.4540 + data['step'] * 0.0003)



import numpy as np 
import pandas as pd
import theano 
from theano import function
import theano.tensor as T
import matplotlib.pyplot as plt 



# set up 
rng = np.random.RandomState(27)

# setup theano
GPU = True
if GPU:
    print("Device set to GPU")
    try: theano.config.device = 'gpu'
    except: pass    # its already set
    theano.config.floatX = 'float32'
else:
    print("Running with CPU")



######################################################################
# network constants
#######################################################################

learning_rate = 0.0001
epochs = 3
n_samples = 1       # set this value in read data
n_hidden = 3   
n_days = 21
n_in = 1
n_out = 1
L1 = 0.001        
L2 = 0.001


#############################################################################
# prep data
#############################################################################


def read_data_file():

    data = pd.read_csv('LeveledLogStockData.csv', parse_dates=True, index_col=0)
    
    # keep only what's necessary
    data = data[['leveled log Nasdaq']]
    data.columns = ['Nasdaq']


    # move all data from -1...1 to 0..2
    data['Nasdaq'] += 1.

    n_samples = len(data)
    # daily has 6805 samples
    

    # quick check on data
    #print(data)
    #plt.figure(figsize=(16,16))
    #plt.plot(data['Nasdaq'], label="Daily Nasdaq")
    #plt.legend(loc='best')
    #plt.title("Log of Nasdaq data rotated to follow x_axis")
    #plt.show()
    
    # how many days ahead to predict?
    # shift 1 for one day predictions, 5 for weekly, 21 monthly, 63 quarterly, 253 yrly

    # pull out one year of data
    n_valid = 253
    
    # x is today's data, shift by 1 so y is next number in sequence
    x = data['Nasdaq'].values
    y = data['Nasdaq'].shift(n_days).values      

    
    # remove as many items as we've shifted from top of array
    x = x[n_days:-1]
    y = y[n_days:-1]

    # split into training and validation sets
    x_train = x[0:len(data)-n_valid]
    y_train = y[0:len(data)-n_valid]
    x_valid = x[len(x_train):-1]
    y_valid = y[len(y_train):-1]
    
    print(len(x_train), len(y_train), len(x_valid), len(y_valid))


    x = np.asarray(x_train.reshape(len(x_train)))
    y = np.asarray(y_train.reshape(len(y_train)))
    x_valid = np.asarray(x_valid.reshape(len(x_valid)))
    y_valid = np.asarray(y_valid.reshape(len(y_valid)))

    return x.astype('float32'), y.astype('float32'), x_valid.astype('float32'), y_valid.astype('float32')

x, y, x_valid, y_valid = read_data_file()




########################################################################
# test
#######################################################################


class FullyConnected:

    def __init__(self):
        
        # set up and initialize weights
        W_in_values = np.random.rand(n_hidden)
        self.W_in = theano.shared(value=W_in_values, name='W_in', borrow=True)

        W_h_values = np.random.rand(n_hidden, n_hidden)
        self.W_h = theano.shared(value=W_h_values, name='W_h', borrow=True)

        W_out_values = np.random.rand(n_hidden)
        self.W_out = theano.shared(value=W_out_values, name='W_out', borrow=True)


        self.parameters = [self.W_in, self.W_h, self.W_out]


        def save_weights():
            np.savez("Sunspot_weights.npz", *[p.get_value() for p in self.parameters])
        self.save_weights = save_weights
   

        # placeholders for data
        X = T.dscalar('X')
        Y = T.dscalar('Y')

        # -------------  feed forward ----------------------------

        # feed input to hidden units
        hidden_units = X * self.W_in 

        # take hidden output and send to other hidden nodes
        hidden_hidden = hidden_units * self.W_h      # node by column of weights

        # input from other hidden nodes
        hidden_out = T.nnet.relu(hidden_hidden.sum(axis=1) ) # sum row of weights

        # out from hidden nodes to output weights
        out = T.nnet.relu(hidden_out * self.W_out)   # hidden node outputs * output weights

        # predicted
        predicted = T.sum(out)                  # sum all incoming 

        # use to see which weight blow up and for regularization if used
        sum_weights_in = T.sum(self.W_in)
        sum_hidden_weights = T.sum(self.W_h)
        sum_weights_out = T.sum(self.W_out)


        # error - regularization
        cost = (predicted - Y) **2 - sum_hidden_weights * L1 - sum_weights_out **2 * L2

        gradients = T.grad(cost, self.parameters)    # derivatives
        updates = [(p, p - learning_rate * g) for p, g in zip(self.parameters, gradients)]


        # training and prediction functions
        #self.weights_op = theano.function(inputs=[], outputs=[sum_weights_in, sum_hidden_weights, sum_weights_out])
        self.predict_op = theano.function(inputs = [X], outputs = predicted)

        self.train_op = theano.function(
                    inputs = [X, Y],
                    outputs = cost,
                    updates = updates
        )


    def train(self, x, y):

        costs = []
        for i in range(epochs):
            
            cost = 0
            predictions = []
            for j in range(len(y)):
                c = self.train_op(x[j], y[j])
                cost += c 
                predictions.append( self.predict_op(x[j]))
                
                

            
            # output cost so user can see training progress
            cost /= len(y)
            print ("Training cost:", i, "cost:", cost * 100., "%")
            costs.append(cost)
            

        # graph to show accuracy progress - cost function should decrease
        plt.figure(figsize=(20,16))
        plt.plot(y, label='Actual')
        plt.plot(predictions, label='Predicted', linewidth=3, alpha=0.4)
        plt.legend(loc='best')
        plt.title("Nasdaq prediction on training data")
        plt.savefig('training_nasdaq.png')
        plt.show()
        


        
        # predictions on validation data
        new_predictions = []
        validation_cost = 0.0
        for k in range(len(x_valid)):
            p = self.predict_op(x_valid[k])
            new_predictions.append(p)
            validation_cost += y_valid[k] - p

        # plot predicitons on unseen data
        plt.figure(figsize=(20,16))
        plt.plot(y_valid[n_days:-1], label='Actual')
        plt.plot(new_predictions, label='Predicted', linewidth=3, alpha=0.4)
        plt.legend(loc='best')
        plt.title("Nasdaq validation data predictions")
        plt.savefig('nasdaq_validatation.png')
        plt.show()
        
        print("Validation cost: ", validation_cost)


        # predictions on hold out data more than one day ahead
        new_predictions = []
        x = x_valid[0]        # prime the loop with out last known value
        hold_out_cost = 0.0
        prediction_length = len(x_valid)
        for k in range(prediction_length):    # number of days ahead to predit
            p = self.predict_op(x)
            x = p               # use today's prediction to figure out tomorrow's value
            new_predictions.append(p)
            hold_out_cost += x_valid[k] - p


        # save p before adjusting data back
        p_hold_out = new_predictions
        #for i in range(prediction_length):
        #    print("x %lf, y %lf, p %lf" % (x_valid[i], y_valid[i], p_hold_out[i]))

        #####################################################################################
        # reverse data from scaled and rotated back to actual
        
        # de-rotate data                     ( reverse this data['log NASDAQ'] - (6.4540 + data['step'] * 0.0003))      
        starting_step = 6805 - prediction_length         #  get starting step

        # de-log data                        ( e ^ x )
        
        for i in range(len(y_valid)):
            y_valid[i] -= 1.                            # recenter between -1..1 from 0..2   ( subtract 1)
            y_valid[i] = y_valid[i] + (6.4540 + (starting_step+i) * 0.0003)      # remove rotation
            y_valid[i] = np.e **y_valid[i]                 # undo log


        for i in range(len(new_predictions)):
            new_predictions[i] -= 1.
            new_predictions[i] = new_predictions[i] + (6.4540 + (starting_step+i) * 0.0003)
            new_predictions[i] = np.e **new_predictions[i]


        # plot predictions on unseen data
        plt.figure(figsize=(20,16))
        plt.plot(y_valid[0: 2 * n_days], label='Actual')
        plt.plot(new_predictions[0:2 * n_days], label='Predicted')
        plt.scatter(n_days, new_predictions[n_days], label='Predicted value on target date', s=50, alpha=0.4)
        plt.scatter(n_days, y_valid[n_days], label='Actual value on target date', s=10, alpha=0.2)
        plt.legend(loc='best')
        t = "Nasdaq prediction in " + str(n_days) + " days"
        plt.title(t)
        plt.savefig('nasdaq_prediction.png')
        plt.show()

        print("hold out cost ", hold_out_cost)
        print("Next week prediction %lf, next week actual %lf" %(y_valid[0], new_predictions[0]))
        



network = FullyConnected()
network.train(x, y)

