


# http://github.com/timestocome
#
# build Markov Chains using daily Nasdaq data
# and use it to predict next few days market movements



from collections import Counter
import pandas as pd
import numpy as np




data_file = 'NASDAQ.csv'



##########################################################################################################
# utility functions
#########################################################################################################
choices = ['HG', 'MG', 'LG', 'S', 'LL', 'ML', 'HL']

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


# read in file from finance.yahoo.com, save dates and log of daily opening price
# and daily change sorted into buckets
def read_data(file):

    # read in data file
    input = pd.read_csv(file, parse_dates=True, index_col=0)

    # convert to log scale and calculate daily change ( volatility )
    input['LogOpen'] = np.log(input['Open'])
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

# build data chain from cleaned data series
def build_data_chain(input):

    # calculate number of times each item appears and save
    counts = input.value_counts().to_dict()
    total = len(input)
    # print(counts)

    
    # convert to probabilty of each change occurring s/b sum to ~1
    probability = {k: v / total for k, v in counts.items()}
    #print(probability)
    
    return probability
    



def build_state_machine(input):
    
    # create a node for every unique letter
    uniques = set(input)

    # get total shifts so we can use probabilities in predictions
    total = len(input)
    
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
    markov = []
    for i in range(len(edge_count)):    
       # g.add_edge(edge_nodes[i][0], edge_nodes[i][1], edge_count[i])
        markov.append((edge_nodes[i][0], edge_nodes[i][1], edge_count[i]/total))    

    return markov

    

############################################################################################################
# make predictions
#
# with enough data you could train this state machine using a forward feed neural network
# and it'd work like the lastest version of Google's Go playing AI
#
# markov machine that was built from data
# n_days how far ahead to predict, it'll get worse the longer this is unless it's trained with a nn
# greedy (most likely ) vs random choice .9 is 10% random, 90% greedy
# prime is the state to start with
############################################################################################################

def make_prediction(state_machine, prime, n_days=5, greedy=.9):

    # prime the loop
    prediction_chain = [prime]     # list of predictions
    n_choices = len(choices)       # number of possiblities
    current_state = prime          # starting condition

    #print('prime', prime)
    
    # loop over state machine picking most likely or random
    for i in range(n_days):
    
        r1 = np.random.random()

        if r1 < greedy:         # pick most likely

            # markov chain is already sorted, most likely option is first
            for j in state_machine:
                if j[0] == current_state:
                    pick = j[1]
                    break
                
                
        else:                  # pick random
            r2 = np.random.randint(n_choices)
            pick = choices[r2]

        current_state = pick
        prediction_chain.append(pick)
            

    return prediction_chain


        
        



    





#######################################################################################################
# build state machine and probabilities from input data
# predictions based on probabilities and state machine
######################################################################################################
data = read_data(data_file)
#print(data)

probabilities = build_data_chain(data)
#print(probabilities)

markov = build_state_machine(data)
#print(markov)


                

# make predictions
current = 'ML' # prime with today's gain or loss bucket
n = 5          # number of days to look ahead
g = .9         # percent of time to pick most likely vs random


predictions = make_prediction(markov, current, n, g)

print('Predicted movements next 5 days starting with today\'s %s, non-random choices %f%% of time %s' %(current, g, predictions))
