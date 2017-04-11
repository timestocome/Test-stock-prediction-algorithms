

# http://github.com/timestocome

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import py_func


import matplotlib.pyplot as plt 


# use clustering to determine best quarter, month, day of week to invest or not be in the market


# no difference in years between days up and down - 
# re-clustering with out years
# only ~67% accurate with all data
# 93% accurate with top 100 peak days

##########################################################################
# load data from  https://www.measuringworth.com/datasets/DJA/index.php
##########################################################################

dja = pd.read_csv('cleaned_dja.csv',  index_col=0)        # 31747 days of data 



########################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# top movement days, comment this out to use all of the data
# top n_peaks gains + top n_peaks loss days
n_peaks = 100

top_gains = dja.nlargest(n_peaks, 'percent_dx')
top_gains['dow'] = pd.DatetimeIndex(top_gains['Date']).dayofweek

top_losses = dja.nsmallest(n_peaks, 'percent_dx')
top_losses['dow'] = pd.DatetimeIndex(top_losses['Date']).dayofweek

print("Top ", n_peaks)
print(top_gains[['Date','dow', 'percent_dx']])

print('-------------------------------------')
print("Bottom ", n_peaks)
print(top_losses[['Date', 'dow', 'percent_dx']])


dja = pd.concat([top_gains, top_losses])
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##########################################################################






n_samples = len(dja)
targets = np.where(dja['percent_dx'] > 0, 1, 0)

#features = [dja.columns.values]
#print(features)

'''
features = ['Q1','Q2', 'Q3', 'Q4', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'M1', 'M2',
       'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'Y0',
       'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10', 'Y11',
       'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17', 'Y18', 'Y19', 'Y20',
       'Y21', 'Y22', 'Y23', 'Y24', 'Y25', 'Y26', 'Y27', 'Y28', 'Y29',
       'Y30', 'Y31', 'Y32', 'Y33', 'Y34', 'Y35', 'Y36', 'Y37', 'Y38',
       'Y39', 'Y40', 'Y41', 'Y42', 'Y43', 'Y44', 'Y45', 'Y46', 'Y47',
       'Y48', 'Y49', 'Y50', 'Y51', 'Y52', 'Y53', 'Y54', 'Y55', 'Y56',
       'Y57', 'Y58', 'Y59', 'Y60', 'Y61', 'Y62', 'Y63', 'Y64', 'Y65',
       'Y66', 'Y67', 'Y68', 'Y69', 'Y70', 'Y71', 'Y72', 'Y73', 'Y74',
       'Y75', 'Y76', 'Y77', 'Y78', 'Y79', 'Y80', 'Y81', 'Y82', 'Y83',
       'Y84', 'Y85', 'Y86', 'Y87', 'Y88', 'Y89', 'Y90', 'Y91', 'Y92',
       'Y93', 'Y94', 'Y95', 'Y96', 'Y97', 'Y98', 'Y99', 'Y100', 'Y101',
       'Y102', 'Y103', 'Y104', 'Y105', 'Y106', 'Y107', 'Y108', 'Y109',
       'Y110', 'Y111', 'Y112', 'Y113', 'Y114', 'Y115', 'Y116', 'Y117']
'''

features = ['Q1','Q2', 'Q3', 'Q4', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'M1', 'M2',
            'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12']

samples = (dja[features]).as_matrix()




k = 2                         # number of clusters
max_iterations = 50

# pick random samples to use as intitial centroids
# also tried picking random values from all centroids 
def initial_cluster_centeroids(X, k): 

    c = []
    for i in range(k):
        z = np.random.randint(n_samples)
        c.append(X[i])

    return c


# assign sample to closest cluster
def assign_to_cluster(X, centroids):  

    expanded_vectors = tf.expand_dims(X, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    mins = tf.argmin(distances, 0)

    return mins
    


def recompute_centroids(X, Y):

    sums = tf.unsorted_segment_sum(X, Y, k)
    counts = tf.unsorted_segment_sum(tf.ones_like(X), Y, k)

    return sums / counts



with tf.Session() as sess:

    # init everything
    tf.global_variables_initializer().run()
    
   

    # cluster 
    #centroids = initial_cluster_centeroids(x, k)
    centroids = initial_cluster_centeroids(samples, k)
    centroids = np.asarray(centroids, dtype='float64')
    tf.cast(centroids, tf.float64)

    x = np.asarray(samples, dtype='float64')
    tf.cast(x, tf.float64)


    i, converged = 0, False 
    while not converged and i < max_iterations:
        i += 1
        y = assign_to_cluster(x, centroids)
        centroids = sess.run(recompute_centroids(x, y))


    # check predictions
    print("Centroids")
    print(centroids)

    print("-------------------------------------------------")
    print("Check groups ")
    correct = len(targets) - sum(targets - y.eval())   # zero if same so adjust
    score = correct / len(targets)
    print("Accuracy ", score)

    # split samples into targets
    group0 = []
    group1 = []
    for i in range(len(x)):
        if targets[i] == 0: group0.append(x[i])
        if targets[i] == 1: group1.append(x[i])



    # sanity check clusters 
    # get average for each value 
    group0 = np.asarray(group0)
    group0_mean = group0.mean(0)

    group1 = np.asarray(group1)
    group1_mean = group1.mean(0)

    print("-------------------------------------------")
    print("Differences between winning days and losing")
    diff = group1_mean - group0_mean

    text = ""
    for i in range(len(features)):
        if diff[i] < 0: text = "Worse"
        else: text = "Better"

        print("Times to invest: %s %s %.4lf" %(text, features[i], diff[i]))

sess.close()
