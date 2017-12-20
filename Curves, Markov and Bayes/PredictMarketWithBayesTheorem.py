


# http://github.com/timestocome
#
# Build Bayesian using daily BitCoin Closing Price
# Use today and tomorrow's data to see if it can predict 
# next few days market movements



from collections import Counter
import pandas as pd
import numpy as np



# http://coindesk.com/price
data_file = 'BitCoin_Daily_Close.csv'







##########################################################################################################
# utility functions
#########################################################################################################
states = ['HG', 'MG', 'LG', 'S', 'LL', 'ML', 'HL']

# break daily changes into buckets
# I'm using 6 buckets HG +10%+, MG +5%-10%, LG +1/2%-5%, S -1/2%-1/2%, LL -1/2%-5%, ML -5%-10%, HL -10%+
# the more buckets (complexity you add the more data you'll need to get good results)
def categorize(gain):

    if gain > 10.:
        return 'HG'
    elif gain > 5.:
        return 'MG'
    elif gain > 0.5:
        return 'LG'
    elif gain > -0.5:
        return 'S'
    elif gain > -5.:
        return 'LL'
    elif gain > -10.:
        return 'ML'
    else:
        return 'HL'


# read in file from http://coindesk.com/price, save dates and log of daily closing price
# and daily change sorted into buckets
def read_data(file):

    # read in data file
    input = pd.read_csv(file, parse_dates=True, index_col=0)

    # convert to log scale and calculate daily change ( volatility )
    input['LogOpen'] = np.log(input['Close'])
    input['change'] = input['LogOpen'] / input['LogOpen'].shift(1)
    input = input.dropna(axis=0)

    # group changes into buckets and label them
    input['change'] = (input['change'] - 1.) * 1000.
    input['dx'] = input['change'].apply(categorize)

    # dump the columns we're done with
    #print(input['dx'].describe())
    return input['dx']

    


########################################################################################################
# build data structures
########################################################################################################

def probability_movement(input):

    # calculate number of times each item appears and save
    counts = input.value_counts().to_dict()
    total = len(input)
    print('Counts: ', counts)

    
    # convert to probabilty of each change occurring s/b sum to ~1
    probability = {k: v / total for k, v in counts.items()}
    print('Probability:', probability)
    
    return probability





# want probability of today, given tomorrow
def probability_today_given_tomorrow(input):

    # create a node for every unique letter
    uniques = set(input)

    # get total shifts so we can use probabilities in predictions
    total = len(input)

    # reverse data need the likelihood of today based on what happens tomorrow using
    # historical data
    print(input)

    
    # create edges
    edges = []
    for i in range(len(uniques)):
        
        n = list(uniques)[i]

        for j in range(len(data)-1):

            if data[j] == n:
                edges.append( (n, data[j+1]) )
                
               
    # count times each edge occurs
    edge_count = pd.Series(data=edges).value_counts()
    edge_nodes = edge_count.index.tolist()

    # add edges to graph
    prob_today_given_tomorrow = []
    for i in range(len(edge_count)):    
       # g.add_edge(edge_nodes[i][0], edge_nodes[i][1], edge_count[i])
        prob_today_given_tomorrow.append((edge_nodes[i][0], edge_nodes[i][1], edge_count[i]/total))    

    return prob_today_given_tomorrow





###########################################################################################################
# make predictions
#
# predict P(tomorrow|today) = P(today|tomorrow) * P(today) / P(tomorrow)
# predict P(A|B) = P(B|A) * P(A) / P(B)
#
# in this case PA = PB since it's just the percent a given gain or loss occurs
############################################################################################################

def make_predictions(PA, PB, PBA):

    print("----------------")
    print('PA')
    print(PA)

    print('----------------')
    print('PBA')
    print(PBA)

    
    # loop over all possibilities
    predictions = []
    for i in range(len(PBA)):

        pba = PBA[i]       # probability today given tomorrow
        pba_s1 = pba[0]    # today
        pba_s2 = pba[1]    # tomorrow
        pba_p = pba[2]     # probability of today's movement

        pa = PA.get(pba[0])   # probability of a given movement
        pb = PB.get(pba[1])   # probablity of a givent movement
        

        # baye's formula
        pab_p = (pba_p * pa) / pb

        predictions.append((pba_s1, pba_s2, pab_p))



    return predictions

       


def get_predictions(s, p):

    predict = []
    total = 0.

    for i in range(len(p)):

        if p[i][0] == s:
            predict.append((p[i][1], p[i][2]))
            total += p[i][2]
           
    return predict, total




#######################################################################################################
# build probabilities from input data
# predictions based on probabilities
######################################################################################################

# read in data 
data  = read_data(data_file)


# calculate probabilities of any movement on a given day
prob = probability_movement(data)

# calculate the probability of today's movement based on tomorrow's using historical data
prob_today_given_tomorrow = probability_today_given_tomorrow(data)
print(prob_today_given_tomorrow)


# now run forward
predictions = make_predictions(prob, prob, prob_today_given_tomorrow)



# print out predictions in a useful format
for s in states:

    p, t = get_predictions(s, predictions)
    
    print('-------------------------')
    print("if today's movement %s , then tomorrow's prediction "% s)
    for i in range(len(p)):
        
        print('Movement:  %s  %.4lf%%' % (p[i][0], p[i][1]/t * 100.))
        
    
    
    
