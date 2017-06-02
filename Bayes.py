# http://github.com/timestocome

# see what, if anything can be predicted with bayes rule
# meh everything is too interconnected to make a good prediction
# accuracy is the same as just using the probability of it being up or down on a given day

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


######################################################################
# data
########################################################################
# read in datafile created in LoadAndMatchDates.py
data = pd.read_csv('StockDataWithVolume.csv', index_col='Date', parse_dates=True)
features = [data.columns.values]

# switch to log data
log_data = np.log(data)

# predict tomorrow's gain/loss given today's information
log_data['target'] = log_data['NASDAQ'].shift(-1)

# indicators of tomorrow's gain or loss
features = ['NASDAQ', 'VIX', 'US GDP', 'Gold', '1yr Treasury', '10yr Bond', 'UnEmploy', 'NASDAQ_Volume']

# log returns ( change in value ) for all features
log_change = log_data - log_data.shift(1)

# clean up 
log_data.dropna()
log_change.dropna()

n_samples = len(log_data)

# increase/decrease each day over previous day for all columns
p_up = (log_change > 0.).astype(int)
p_down = (log_change < 0.).astype(int)



##############################################################################################
predict_feature_target = [0.0, 0.0, 0.0, 0.0]      # 4 probabilities per feature 00,01,10,11 
feature_prediction = []             # one prediction array per feature
for f in features:

    # bayes probabilty stock up tomorrow given today's feature increase or decrease
    # probability (B|A) = (p(B) * p(A|B)) / ( p(B) * p(A|B) + p(notB) * p(A|notB)
    # probability (Stock up | Indicator up ) = (p(Stock up) * p(Indicator up | Stock up) / (p(Stock up) * p(Indicator up|Stock up) + p(Stock down) * p(Indicator up | Stock down))
    
    p_stock_up = p_up.sum()['target'] / n_samples       # p(B)
    p_stock_down = p_down.sum()['target'] / n_samples   # p(not B)


    p_stock_up_feature_up = ((p_up['target'] + p_up[f]).value_counts()[2]) / n_samples      # p(A|B)
    p_stock_up_feature_down = ((p_up['target'] + p_down[f]).value_counts()[2]) / n_samples  # p(A|notB)
    p_stock_d_feature_up = ((p_down['target'] + p_up[f]).value_counts()[2]) / n_samples     # p(notA|B)
    p_stock_d_feature_down = ((p_down['target'] + p_down[f]).value_counts()[2]) / n_samples # p(notA|notB)

    # check both s/b ~ 1.
    # total_a = p_stock_up + p_stock_down
    # total_b = p_stock_up_feature_up + p_stock_up_feature_down + p_stock_d_feature_up + p_stock_d_feature_down


    # bayes
    p_target_up_feature_up = (p_stock_up * p_stock_up_feature_up) / (p_stock_up * p_stock_up_feature_up + p_stock_down * p_stock_d_feature_up)
    p_target_up_feature_down = (p_stock_up * p_stock_up_feature_down) / (p_stock_up * p_stock_up_feature_down + p_stock_down * p_stock_d_feature_down)
    p_target_down_feature_up = (p_stock_down * p_stock_d_feature_up) / (p_stock_down * p_stock_d_feature_up + p_stock_up * p_stock_up_feature_up)
    p_target_down_feature_down = (p_stock_down * p_stock_d_feature_down) / (p_stock_down * p_stock_d_feature_down + p_stock_up * p_stock_up_feature_down)

    # check sums of probabilities
    #total_c = p_target_up_feature_up + p_target_up_feature_down + p_target_down_feature_up + p_target_down_feature_down
    #print(p_target_up_feature_up, p_target_up_feature_down, p_target_down_feature_up, p_target_down_feature_down, total_c)


    
    print("Probability Nasdaq up tomorrow if %s up today: %f" % (f, p_target_up_feature_up))
    print("Probability Nasdaq down tomorrow if %s up today: %f" % (f, p_target_down_feature_up))
    print("Probability Nasdaq up tomorrow if %s down today: %f" % (f, p_target_up_feature_down))
    print("Probability Nasdaq down tomorrow if %s down today: %f" %  (f, p_target_down_feature_down))
    print("--------------------------------------------------------------------------------------------")

    predict_feature_target = [p_target_down_feature_down, p_target_down_feature_up, p_target_up_feature_down, p_target_up_feature_up]
    feature_prediction.append(predict_feature_target)

print("Probability NASDAQ up tomorrow %f, down tomorrow %f" %(p_stock_up, p_stock_down))
##############################################################################################
# since these features are not linearly independent stringing them together is likely to 
# reduce rather than increase the information.

# test each predictor(feature) against target to check accuracy
for f in range(len(features)):

    correct = 0
    f_predict = feature_prediction[f]       # get probability array for this feature

    print("**********************************************************************")
    print("Feature ", features[f])
    print("Probabilities", f_predict)


    days_up = 0
    days_down = 0
    for d in range(len(p_up)):              # for each day in our data array
        today = p_up.ix[d]
        next_day = -1

        if today[f] == 1:           # indicator is up today
            days_up += 1
            up = f_predict[3]       # feature up stock up probability
            down = f_predict[1]     # feature up stock down
            if up > down:   next_day = 1
            else: next_day = 0

        else:                       # indicator is down today
            days_down += 1
            up = f_predict[2]       # feature down, stock up
            down = f_predict[0]     # feature down, stock down
            if up > down: next_day = 1
            else: next_day = 0
    
        if today['target'] == next_day: correct += 1
    
    print("Feature days up %d, down %d" % (days_up, days_down))
    print("Feature: %s, accuracy %f" % (features[f], correct/len(p_up) * 100.))

##################################################################################################
# output
##################################################################################################
'''
Probability Nasdaq up tomorrow if NASDAQ up today: 0.609308
Probability Nasdaq down tomorrow if NASDAQ up today: 0.390692
Probability Nasdaq up tomorrow if NASDAQ down today: 0.579626
Probability Nasdaq down tomorrow if NASDAQ down today: 0.420374
--------------------------------------------------------------------------------------------
Probability Nasdaq up tomorrow if VIX up today: 0.574110
Probability Nasdaq down tomorrow if VIX up today: 0.425890
Probability Nasdaq up tomorrow if VIX down today: 0.615124
Probability Nasdaq down tomorrow if VIX down today: 0.384876
--------------------------------------------------------------------------------------------
Probability Nasdaq up tomorrow if US GDP up today: 0.586295
Probability Nasdaq down tomorrow if US GDP up today: 0.413705
Probability Nasdaq up tomorrow if US GDP down today: 0.618268
Probability Nasdaq down tomorrow if US GDP down today: 0.381732
--------------------------------------------------------------------------------------------
Probability Nasdaq up tomorrow if Gold up today: 0.592722
Probability Nasdaq down tomorrow if Gold up today: 0.407278
Probability Nasdaq up tomorrow if Gold down today: 0.599811
Probability Nasdaq down tomorrow if Gold down today: 0.400189
--------------------------------------------------------------------------------------------
Probability Nasdaq up tomorrow if 1yr Treasury up today: 0.608437
Probability Nasdaq down tomorrow if 1yr Treasury up today: 0.391563
Probability Nasdaq up tomorrow if 1yr Treasury down today: 0.577550
Probability Nasdaq down tomorrow if 1yr Treasury down today: 0.422450
--------------------------------------------------------------------------------------------
Probability Nasdaq up tomorrow if 10yr Bond up today: 0.584990
Probability Nasdaq down tomorrow if 10yr Bond up today: 0.415010
Probability Nasdaq up tomorrow if 10yr Bond down today: 0.603238
Probability Nasdaq down tomorrow if 10yr Bond down today: 0.396762
--------------------------------------------------------------------------------------------
Probability Nasdaq up tomorrow if UnEmploy up today: 0.529718
Probability Nasdaq down tomorrow if UnEmploy up today: 0.470282
Probability Nasdaq up tomorrow if UnEmploy down today: 0.583892
Probability Nasdaq down tomorrow if UnEmploy down today: 0.416108
--------------------------------------------------------------------------------------------
Probability Nasdaq up tomorrow if NASDAQ_Volume up today: 0.592700
Probability Nasdaq down tomorrow if NASDAQ_Volume up today: 0.407300
Probability Nasdaq up tomorrow if NASDAQ_Volume down today: 0.599805
Probability Nasdaq down tomorrow if NASDAQ_Volume down today: 0.400195
--------------------------------------------------------------------------------------------
Probability NASDAQ up tomorrow 0.547832, down tomorrow 0.450992
**********************************************************************
Feature  NASDAQ
Probabilities [0.42037390530311558, 0.3906923454634732, 0.57962609469688453, 0.6093076545365268]
Feature days up 3609, down 3196
Feature: NASDAQ, accuracy 54.783248
**********************************************************************
Feature  VIX
Probabilities [0.38487565522111888, 0.42589026197209323, 0.615124344778881, 0.57410973802790677]
Feature days up 3630, down 3175
Feature: VIX, accuracy 54.783248
**********************************************************************
Feature  US GDP
Probabilities [0.38173224428873503, 0.41370478544147382, 0.61826775571126502, 0.58629521455852618]
Feature days up 3736, down 3069
Feature: US GDP, accuracy 54.783248
**********************************************************************
Feature  Gold
Probabilities [0.4001892863903086, 0.40727848637730329, 0.5998107136096914, 0.59272151362269676]
Feature days up 3729, down 3076
Feature: Gold, accuracy 54.783248
**********************************************************************
Feature  1yr Treasury
Probabilities [0.42245022748793365, 0.39156335064496012, 0.5775497725120663, 0.60843664935503983]
Feature days up 3232, down 3573
Feature: 1yr Treasury, accuracy 54.783248
**********************************************************************
Feature  10yr Bond
Probabilities [0.39676242530566913, 0.41500975891397984, 0.60323757469433081, 0.58499024108602016]
Feature days up 39, down 6766
Feature: 10yr Bond, accuracy 54.783248
**********************************************************************
Feature  UnEmploy
Probabilities [0.41610835526838685, 0.47028192676423636, 0.58389164473161304, 0.5297180732357637]
Feature days up 3412, down 3393
Feature: UnEmploy, accuracy 54.783248
**********************************************************************
Feature  NASDAQ_Volume
Probabilities [0.40019451255665861, 0.40730023656369241, 0.59980548744334139, 0.59269976343630759]
Feature days up 2695, down 4110
Feature: NASDAQ_Volume, accuracy 54.783248
'''