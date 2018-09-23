#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:23:07 2018


# http://github.com/timestocome

# try recurrent networks from 'Deep Learning with Python' on BTC
# from section 6 in book

# really disappointed the 1D convolutionals were so bad, I had high hopes for them

"""

# bitcoin data from finance.yahoo.com
# btc.csv


###############################################################################
# process data
###############################################################################

import pandas as pd
import numpy as np




##############################################################################
# this data should be rotated to x axis and log differences used but 
# I'm just using the BitCoin data to check my understanding of Keras 
##############################################################################

# display more than the default data on screeen
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)



def print_info(data):
    print('------------------------------------------------------------')
    print(data.head())
    print(data.describe())
    print('Total rows', len(data))
    print('Rows with bad data', data.isnull().sum())




# read in csv file downl0oaded from finance.yahaoo
btc_file = 'btc.csv'
data = pd.read_csv(btc_file, parse_dates=True)

# make sure file read in okay
#print_info(data)




# keep daily Opne, Volume
data = data[['Open', 'Volume', 'High', 'Low', 'Close', 'Volume']]



# save to rescale output
mean = data['Open'].mean()
std = data['Open'].std()
def rescale(output):
    
    output = (std * output) + mean

    return output





# normalize data
data['Open'] -= data['Open'].mean()
data['High'] -= data['High'].mean()
data['Low'] -= data['Low'].mean()
data['Close'] -= data['Close'].mean()
data['Volume'] -= data['Volume'].mean()


data['Open'] /= data['Open'].std()
data['High'] /= data['High'].std()
data['Low'] /= data['Low'].std()
data['Close'] /= data['Close'].std()
data['Volume'] /= data['Volume'].std()



# convert to numpy array
data = data.values

# something flaky with Spider in Anaconda 
# it seems to remember old vars and then randomly forget them
# idk - cheap hack fix
float_data = data



#print(data)
#print(data.shape)

# divy things up
n_features = data.shape[1]
n_samples = data.shape[0]           # total rows of data
n_test = n_samples // 10
n_validate = n_test
n_train = n_samples - n_test - n_validate





# convert to proper format for an RNN type network
# see section 6.3 in book
def generator(data, lookback, delay, min_index, max_index, batch_size, step, shuffle=False):

    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    
    while 1:
        
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            
            rows = np.arange(i, min(i + batch_size, max_index))
            i + len(rows)
            
        samples = np.zeros((len(rows), 
                            lookback // step, 
                            data.shape[-1]))
    
        targets = np.zeros((len(rows),))
        
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
            
        yield samples, targets
        
        
        
   
lookback = 20       # days worth of data to consider
step = 1            # one data point per day
delay = 1           # 1 day == tomorrow's price, 2 days == day after tomorrow....  
batch_size = 30     # days


train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=n_train,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)


val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=n_train + 1,
                    max_index=n_train + n_validate,
                    step=step,
                    batch_size=batch_size)


test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=n_train + n_validate + 1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size
                     )


val_steps = (n_validate - 1 - lookback)
test_steps = val_steps
    
print('val steps', val_steps)
print('test steps', test_steps)  

##############################################################################
# couldn't find or figure out how to use the test generator to get date to feed 
# into model_predict - all examples use numpy x, y
# so this will have to do until I figure out how Keras does it
# yach ( yet another cheap hack )
###############################################################################
def prediction_data(data, lookback, delay, min_idx, max_idx, step, batch_size):
    
        max_index = len(data) - delay - 1
        i = min_idx + lookback
        
 
        if i + batch_size >= max_index:
            i = min_index + lookback
            
        rows = np.arange(i, max(i + batch_size, max_index))
        i + len(rows)
            
        samples = np.zeros((len(rows), 
                            lookback // step, 
                            data.shape[-1]))
    
        targets = np.zeros((len(rows),))
        
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
            
       
        
        return samples, targets
        



hold_out_data = data[-n_test:]
x, y = prediction_data(hold_out_data,
                       lookback=lookback,
                       delay=delay,
                       min_idx=0,
                       max_idx=None,
                       step=step,
                       batch_size=1)


'''
###############################################################################
# test Sequential model
# run a quick check to make sure data processing okay
# and get a quick baseline
# validation loss bouncy, ends ~ 2 ( * std to undo scaling )
# so that useless ( as expected )
###############################################################################
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop




model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_gen,
                              steps_per_epoch=20,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

model.save_weights('sequential_model.h5')

predictions = model.predict(x)


predictions = rescale(predictions)
y = rescale(y)



'''
'''
###############################################################################
# test GRU model
# slightly better with same steps, epochs
# still a wide variance on the validation data but doubling epochs to 100 helps
# adding drop out increased the validation error, 
# doubling steps per epoch made validation bounce more - over fitting
# added GRU 64 layer and increased dropout - meh, not any better
# decreased GRU layer size by half -
# results are still pitiful but better than the last model
###############################################################################

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop



model = Sequential()

model.add(layers.GRU(32, 
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, float_data.shape[-1])))

model.add(layers.GRU(32, 
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     activation='relu'))

model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_gen,
                              steps_per_epoch=200,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)

model.save_weights('gru_model.h5')

# test model on hold out data
predictions = model.predict(x)

predictions = rescale(predictions)
y = rescale(y)

'''
'''
###############################################################################
# test bi-directional RNN
# totally overfit after 20 expochs, otherwise about the same  or slightly better
# performance as previous models
###############################################################################
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop








model = Sequential()

model.add(layers.Bidirectional(layers.GRU(32), 
                               input_shape=(None, float_data.shape[-1])))

model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_gen,
                              steps_per_epoch=32,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)

model.save_weights('bi_direction_rnn.h5')


# test model on hold out data
predictions = model.predict(x)

predictions = rescale(predictions)
y = rescale(y)

'''
'''
###############################################################################
# let's checkout 1D covnets
# ugh that was terrible
###############################################################################
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop


filter_size = 5

model = Sequential()


model.add(layers.Conv1D(64, filter_size, 
                        activation='relu',
                        input_shape=(None, data.shape[-1])))
model.add(layers.MaxPooling1D(filter_size-2))


model.add(layers.Conv1D(64, filter_size, activation='relu'))
model.add(layers.GlobalMaxPooling1D())



model.add(layers.Dense(1))


model.compile(optimizer=RMSprop(), loss='mae')


history = model.fit_generator(train_gen,
                              steps_per_epoch=100,
                              epochs=100,
                              validation_data=val_gen,
                              validation_steps=val_steps)

model.save_weights('1d_conv.h5')


# test model on hold out data
predictions = model.predict(x)

predictions = rescale(predictions)
y = rescale(y)

'''
###############################################################################
# let's checkout 1D covnet with gru layer
# ugh that was terrible too
###############################################################################
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop


filter_size = 5

model = Sequential()

model.add(layers.Conv1D(64, filter_size, 
                        activation='relu',
                        input_shape=(None, data.shape[-1])))
model.add(layers.MaxPooling1D(filter_size-2))


model.add(layers.Conv1D(64, filter_size, activation='relu'))
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))


model.compile(optimizer=RMSprop(), loss='mae')


history = model.fit_generator(train_gen,
                              steps_per_epoch=100,
                              epochs=100,
                              validation_data=val_gen,
                              validation_steps=val_steps)


model.save_weights('1d_conv_gru.h5')


# test model on hold out data
predictions = model.predict(x)

predictions = rescale(predictions)
y = rescale(y)

###############################################################################
# plots
###############################################################################
import matplotlib.pyplot as plt



# plot predictions
plt.figure(figsize=(16,16))

temp = float_data[:, 1] 
plt.plot(range(len(predictions)), predictions, label="Predictions")
plt.plot(range(len(predictions)), y, label="Actual")
plt.title('Predictions vs Actual')
plt.legend()
plt.savefig('1d_conv__gru_predictions.png')
plt.show()





# plot training and validation losses

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure(figsize=(16,16))

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.savefig('1d_conv_gru_loss.png')
plt.legend()

plt.show()






