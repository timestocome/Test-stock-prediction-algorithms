
# https:github.com/timestocome
# take xor neat-python example and convert it to predict  tomorrow's stock 
#       market change using last 5 days data

# uses Python NEAT library
# https://github.com/CodeReclaimers/neat-python


from __future__ import print_function
import os
import neat
import visualize


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##############################################################################
# stock data previously input, loaded and log leveled
# using LoadAndMatchDates.py, LevelData.py
# reading output LeveledLogStockData.csv
# pick one, any one and use it as input/output
###############################################################################

# read data in
data = pd.read_csv('LeveledLogStockData.csv')
#print(data.columns)
#print(data)

# select an index to use to train network
index = data['leveled log Nasdaq'].values
n_samples = len(index)

# split into inputs and outputs
n_inputs = 5    # number of days to use as input
n_outputs = 1   # predict next day

x = []
y = []
for i in range(n_samples - n_inputs - 1):
    
    x.append(index[i:i+n_inputs] )
    y.append([index[i+n_inputs+1]])
    
x = np.asarray(x)
y = np.asarray(y)    

#print(x.shape, y.shape)
 
# hold out last samples for testing
n_train = int(n_samples * .9)
n_test = n_samples - n_train

print('train, test', n_train, n_test)

train_x = x[0:n_train]
test_x = x[n_train:-1]
train_y = y[0:n_train]
test_y = y[n_train:-1]
print('data split', train_x.shape, train_y.shape)
print('data split', test_x.shape, test_y.shape)




###############################################################################
#  some of these need to be updated in the config-feedforward file
# fitness_threshold = n_train - 1
# num_inputs => n_inputs
# num_hidden => ? how many hidden nodes do you want?
# num_outputs => n_outputs
# 
# optional changes
# population size, activation function, .... others as needed
###############################################################################



n_generations = 20
n_evaluate = 1





def eval_genomes(genomes, config):
    
    for genome_id, genome in genomes:
        genome.fitness = n_train - 1
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    
        for xi, xo in zip(train_x, train_y):
            output = net.activate(xi)
            error = (output[0] - xo[0])**2
            if error < 8:
                genome.fitness -= error
            else:
                genome.fitness -= 8.



def run(config_file):
    
    
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)


    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)


    
    # Add a stdout reporter to show progress in the terminal.
    # True == show all species, False == don't show species
    p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    


    # Stop running after n=n_generations
    # if n=None runs until solution is found
    winner = p.run(eval_genomes, n=n_generations)



    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))



    
    # Show output of the most fit genome against testing data.
    print('\nTest Output, Actual, Diff:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    predicted = []
    for xi, xo in zip(test_x, test_y):
        output = winner_net.activate(xi)
        predicted.append(output)
    
    
    node_names = {-1:'4', -2: '3', -3: '2', -4: '1', -5: 'Yesterday', 0:'Predict Change'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


    # ? save?
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint')
    #p.run(eval_genomes, n_evaluate)


    # plot predictions vs actual
    plt.plot(test_y, 'g', label='Actual')
    plt.plot(predicted,  'r-', label='Predicted')
    plt.title('Test Data')
    plt.legend()
    plt.show()



if __name__ == '__main__':
   
    # find and load configuation file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
