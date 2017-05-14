# http://github.com/timestocome



# break data in to weeks
# cluster weeks
# look for outliers ( weeks that are the highest distance from any centroid )
# see if that gives clues as to good and bad weeks. 

# While it does pull out some weeks as anomalies, it's not predictive
# Bad weeks and good weeks do tend to be different distances on average
# than the average week. Good weeks are closer to the center, bad weeks 
# futher out.


import pandas as pd 
import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt



# pandas display options
pd.options.display.max_rows = 100
pd.options.display.max_columns = 25
pd.options.display.width = 1000

# read in datafile created in LoadAndMatchDates.py
data = pd.read_csv('StockData.csv', index_col='Date', parse_dates=True)


# add daily changes for dj, sp, russell, nasdaq, vix, gold, 
data['djia_dx'] = (data['DJIA'] - data['DJIA'].shift(1)) / data['DJIA']
data['sp_dx'] = (data['S&P'] - data['S&P'].shift(1)) / data['S&P']
data['russell_dx'] = (data['Russell 2000'] - data['Russell 2000'].shift(1)) / data['Russell 2000']
data['nasdaq_dx'] = (data['NASDAQ'] - data['NASDAQ'].shift(1)) / data['NASDAQ']





# downsample from days to weeks, now have ~1400 data samples
weekly_data = data.resample('W-SUN').sum()


# scale data
scale_max = 1
scale_min = -1
weekly_data = (weekly_data - weekly_data.min()) / (weekly_data.max() - weekly_data.min()) * (scale_max - scale_min) + scale_min
#print(weekly_data)




#############################################################################
# peak NASDAQ gain/losses in a week
#############################################################################
n_peaks = 100
top_gains = weekly_data.nlargest(n_peaks, 'nasdaq_dx')
top_losses = weekly_data.nsmallest(n_peaks, 'nasdaq_dx')


'''
print("top gains --------------------------------------------------------------------------------")
print(top_gains[['nasdaq_dx', 'cluster', 'distance']])
print("top losses -------------------------------------------------------------------------------")
print(top_losses[['nasdaq_dx', 'cluster', 'distance']])
'''

#############################################################################################################
# what's different about crash, bottom out weeks, see if they show as outliers
#########################################################################################################

from sklearn.cluster import KMeans


# set up distance calculation
def distance_from_centroid(i, c):
        return np.sqrt(i * i - c * c)
weekly_data['distance'] = 0.


# features to use in clustering
features = ['DJIA', 'S&P', 'Russell 2000', 'VIX', 'US GDP', 'Gold', '1yr Treasury','10yr Bond', 'Real GDP', 'UnEmploy', 'djia_dx', 'sp_dx', 'russell_dx']



# test varying numbers of clusters
def compute_clusters(n):
    n_clusters = n
    n_init = 10

    features = ['DJIA', 'S&P', 'Russell 2000', 'VIX', 'US GDP', 'Gold', '1yr Treasury','10yr Bond', 'Real GDP', 'UnEmploy', 'djia_dx', 'sp_dx', 'russell_dx']

    kmeans = KMeans(n_clusters=n_clusters, random_state=27).fit(weekly_data[features])
    weekly_data['cluster'] = kmeans.labels_
    centroids = kmeans.cluster_centers_
   
    d = []
    for index, row in weekly_data.iterrows():

        c = int(row['cluster'])
        center = centroids[c]
        location = row[['DJIA', 'S&P', 'Russell 2000', 'VIX', 'US GDP', 'Gold', '1yr Treasury', '10yr Bond', 'Real GDP', 'UnEmploy', 'djia_dx', 'sp_dx', 'russell_dx']]
        d.append(pd.np.linalg.norm(center - location))

    weekly_data['distance'] = d





max_value = 0
min_value = 0
for n in range(1,14):

    compute_clusters(n)

    # peak NASDAQ gain/losses in a week
    n_peaks = 100
    top_gains = weekly_data.nlargest(n_peaks, 'nasdaq_dx')
    top_losses = weekly_data.nsmallest(n_peaks, 'nasdaq_dx')



    top_distance = top_gains.distance.mean() / weekly_data.distance.mean()
    bottom_distance = top_losses.distance.mean()/ weekly_data.distance.mean()

    if top_distance > max_value:
        print('Number of clusters', n)
        print('.......Best weeks distance from centroid', top_gains.distance.mean() / weekly_data.distance.mean())
        max_value = top_distance
    
    if bottom_distance > min_value:
        print('Number of clusters', n)  
        print('Worst weeks distance from centroid', top_losses.distance.mean()/ weekly_data.distance.mean())
        min_value = bottom_distance


# best number of clusters = 10
compute_clusters(10)


# plot
plt.figure(figsize=(18, 18))
plt.scatter(weekly_data['cluster'], weekly_data['nasdaq_dx'], s=100., c=weekly_data['distance'], alpha=0.5)
plt.ylabel("Nasdaq return")
plt.xlabel("Clusters")
plt.title("Nasdaq returns")
plt.show()
