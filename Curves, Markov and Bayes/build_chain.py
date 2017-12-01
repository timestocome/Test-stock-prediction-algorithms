


# http://github.com/timestocome
#
# test simple building of state machine to use in markov chain
#



from collections import Counter
import pandas as pd
import numpy as np



    
if __name__ == '__main__':

    
    data = ('H', 'E', 'L', 'L', 'O', 'Sp', 'W', 'O', 'R', 'L', 'D', 'Sp', 'H', 'E', 'L', 'L', 'O', '!')
    print(data)


    # create a node for every unique letter
    uniques = set(data)

  
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
        markov.append((edge_nodes[i][0], edge_nodes[i][1], edge_count[i]))


    print(markov)

        
   
    '''
    # test path
    test_char = 'O'

    print("test path ", test_char)
    for i in markov:

        if test_char == i[0]:
            print(i[0], i[1], i[2])
    

    test_char = 'L'
    print('test path', test_char)
    for i in markov:

        if test_char == i[0]:
            print(i[0], i[1], i[2])

    '''



    chain = []
    markov_path = []
    chain.append('L')
    
    for i in range(5):

        prime_char = chain[i]
        markov_path.append(prime_char)
        print("Loop ", i)
        print("Char ", prime_char)
        
        
        connections = []
        scores = []
        for i in markov:
            if prime_char == i[0]:
                connections.append(i)
                scores.append(i[2])

        
        # pick connection greedy for testing
        #best = scores.index(max(scores))
        #chain.append(connections[best][1])

        # pick random for testing
        best = np.random.randint(0, len(connections))
        chain.append(connections[best][1])
        
        
    print("path", markov_path)
