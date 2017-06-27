# http://github.com/timestocome


import numpy as np 
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt 

# attempt to use an auto-encoder to find deviations from norm 
# meh - there are easier, better ways to do this

###################################################################################
# read in and set up data
# read in nasdaq 
data = pd.read_csv('data/GOLD.csv', parse_dates=True, index_col='Date')
data['Open'] = pd.to_numeric(data['Open'], errors='coerce')

# convert to log scale
data['LogOpen'] = np.log(data[['Open']]) 
data['dx'] = data['LogOpen'] - data['LogOpen'].shift(1)

data = data.dropna()

# only use dates we have volume values for
data = data.loc[data.index >= '12-31-1984']

# scale data
data['LogOpen'] = data['LogOpen'] - data['LogOpen'].min()
data['dx'] = data['dx'] - data['dx'].min()


# check everything looks okay
# print(data)

# need to break into input and output data samples
#data = data.resample('B').bfill()       # fill in missing days ( holidays..)
data = data.resample('B').bfill()       # fill in missing days ( holidays..)


weeks_df = [g for n, g in data.groupby(pd.TimeGrouper('W'))]
months_df = [g for n, g in data.groupby(pd.TimeGrouper('M'))]

print("data ", len(data))
print("weeks ", len(weeks_df))
print("months ", len(months_df))

# see if everything looks okay
# print(weeks)
# print("mins", data['LogOpen'].min(), data['dx'].min(), data['LogVolume'].min())
# print("maxs", data['LogOpen'].max(), data['dx'].max(), data['LogVolume'].max())




# convert to numpy matrix
print(len(weeks_df))
weeks = []
for i in range(len(weeks_df)):
    row = weeks_df[i]
    x = np.asarray(row['LogOpen'].values)
    weeks.append(x)

w = np.asarray(weeks)




#########################################################################
# build an autoencoder


def get_batch(x, size):
    a = np.random.choice(len(x), size, replace=False)
    return x[a]



class Autoencoder:

    def __init__(self, n_input, n_hidden, epoch=100, learning_rate=0.001):

        
        x = tf.placeholder(dtype=tf.float32, shape=[None, n_input])

        with tf.name_scope('encode'):
            weights = tf.Variable(tf.random_normal([n_input, n_hidden], dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([n_hidden]), name='biases')
            encoded = tf.nn.tanh(tf.matmul(x, weights) + biases)

        with tf.name_scope('decode'):
            weights = tf.Variable(tf.random_normal([n_hidden, n_input], dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([n_input]), name='biases')
            decoded = tf.matmul(encoded, weights) + biases 

        
        self.epoch = epoch 
        self.learning_rate = learning_rate
        self.x = x
        self.encoded = encoded 
        self.decoded = decoded 
        
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.x, self.decoded))))
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()



   

    # batch training
    def train(self, data, batch_size=2):

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for i in range(self.epoch):
                for j in range(len(data)):
                    batch_data = get_batch(data, batch_size)
                    l, _ = sess.run([self.loss, self.train_op], {self.x: batch_data})

                if i % 10 == 0:
                    print('Epoch {0}: loss = {1}'.format(i, l))
            self.saver.save(sess, './model.ckpt')
    



    def test(self, data):

        with tf.Session() as sess:
            self.saver.restore(sess, './model.ckpt')
            hidden, reconstructed = sess.run([self.encoded, self.decoded], {self.x: data})
            #print('input', data)
            #print('compressed', hidden)
            #print('reconstructed', reconstructed)

            return reconstructed


#################################################################################################

# train auto encoder

n_hidden = 3
n_input = len(w[0])
ae = Autoencoder(n_input, n_hidden)
ae.train(w)



print("test")
print(w[0], w[0].shape)
print( ae.test([w[0]]) )


errors = []
weeks = []
total = []
for i in range(len(w)):
    
    # get date for this week
    row = weeks_df[i]
    weeks.append( row.index[0] )


    # test this week
    z_in = w[i] 
    z_out = ae.test([w[i]])
    errors.append( np.sum(z_in - z_out) )




#print(errors)
#print(weeks)

##################################################################################################

# plotting
# get nasdaq values for plotting
index = []
for i in range(len(weeks_df)):
    row = weeks_df[i]
    index.append(row['LogOpen'][0]) 

# scale up for plot
errors = np.asarray(errors) * 3 

plt.figure(figsize=(16,16))
plt.plot(errors, label='Deviation from expected', alpha=0.4, lw=2)
plt.plot(index, label='Gold')
plt.legend(loc='best')
plt.title("Look for anomalies in Gold using an auto encoder")
plt.grid('on')

plt.savefig("gold_autoencoder.png")
plt.show()

