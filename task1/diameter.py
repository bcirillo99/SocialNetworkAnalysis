import sys
# adding Folder_2 to the system path
sys.path.insert(0, '../')
import networkx as nx
import math
import itertools as it
from joblib import Parallel, delayed
import time
from utils import *
from argparse import ArgumentParser
import pandas as pd
import random
from tqdm import tqdm
import numpy

#DIAMETER
#Classical algorithm: if runs a BFS for each node, and returns the height of the tallest BFS tree
#It is has computational complexity O(n*m)
#It require to keep in memory the full set of nodes (it may huge)
#It can be optimized by
#1) sampling only a subset of nodes on which the BFS is run (solution may be not exact, quality of the solution depends on the number of sampled nodes)
#2) parallelizing BFSs (depends on the available processing power)
#3) ad-hoc optimization
def diameter(G, sample=None):
    nodes=G.nodes()
    n = len(nodes)
    diam = 0
    if sample is None:
        sample = nodes

    for u in sample:
        udiam=0
        clevel=[u]
        visited=set(u)
        while len(clevel) > 0:
            nlevel=[]
            for c in clevel:
                for v in G[c]:
                    if v not in visited:
                        visited.add(v)
                        nlevel.append(v)
            clevel = nlevel
            udiam += 1
        if udiam > diam:
            diam = udiam

    return diam-1

#PARALLEL IMPLEMENTATION
#Utility used for split a vector data in chunks of the given size.
def chunks(data, size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in it.islice(idata, size)}

#Parallel implementation of diam with joblib
def parallel_diam(G,j):
    diam = 0
    # Initialize the class Parallel with the number of available process
    with Parallel(n_jobs=j) as parallel:
        #Run in parallel diam function on each processor by passing to each processor only the subset of nodes on which it works
        result=parallel(delayed(diameter)(G, X) for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))
        #Aggregates the results
        diam = max(result)
    return diam

#AD-HOC OPTIMIZATION
#This algorithm only returns an approximation of the diameter.
#It explores all edges multiple times for a number of steps that is approximately the diam of the graph.
#Thus, it has computational complexity O(diam*m), that is usually much faster than the complexity of previous algorithm.
#
#The algorithm need to keep in memory a number for each node.
#
#The main idea is the following: at step 1, a node can visit its neighbor, at step 2, a node can visit neighbors of neighbors, and so on,
#until at step=diam it visited entire network. However, running this algorithm requires to save for each node a list of visited nodes,
#that is a too large. Hence, I can keep for each node only the size of this set: at step 1 this corresponds to the degree of node u,
#at step 2 we add the degree of all neighbors of u, and so on. Clearly, this is not precise since we are not considering intersections among neighborhoods,
#but it allow to save only a number for each node. The problem is that this number goes to n, and n may be very large.
#However, we only need to understand if at step i the node u needs a further step to visit other nodes or not.
#To this aim, it is sufficient to keep an estimate of the number of new nodes that u and her neighbors visited at i-th step.
#If one of the neighbors of u has visited at i-th step at least one node more than the one visited by u, then u needs one more step for visiting this new node.
#This is still less precise than keeping an estimate of the visited nodes, but it allows to save a number that is at most the maximum degree.
#Whenever, the maximum degree can still be a very large number. We can reduce the amount of used memory even more.
#we can use an hash function of the degree and simply evaluate if they are different. This requires to save only few bits,
#but it is even more imprecise because collisions may occur.
def stream_diam(G):
    step = 0
    # At the beginning, R contains for each vertex v the number of nodes that can be reached from v in one step
    R={v:G.degree(v) for v in G.nodes()}
    done = False

    while not done:
        done = True
        for edge in G.edges():
            # At the i-th iteration, we change the value of R if there is at least one node that may be reached from v in i steps but not in i steps
            # I realize that this is the case, because I have a neighbor that in i-1 steps is able to visit a number of vertices different from how many I am able to visit
            if R[edge[0]] != R[edge[1]]:
                R[edge[0]] = max(R[edge[0]],R[edge[1]])
                R[edge[1]] = R[edge[0]]
                done = False
        step += 1

    return step

def GenWS2DG(n, r, k, q):
    G = nx.Graph()
    nodes=dict() #This will be used to associate to each node its coordinates
    prob=dict() #Keeps for each pair of nodes (u,v) the term 1/dist(u,v)**q

    # dim is the dimension of the area in which we assume nodes are placed.
    # Here, we assume that the 2D area has dimension sqrt(n) x sqrt(n).
    # Anyway, one may consider a larger or a smaller area.
    # E.g., if dim = 1 we assume that all features of a nodes are within [0,1].
    # However, recall that the radius r given in input must be in the same order of magnitude as the size of the area
    # (e.g., you cannot consider the area as being a unit square, and consider a radius 2, otherwise there will be an edge between each pair of nodes)
    # Hence, the choice of larger values for dim can be preferred if one want to represent r as an integer and not a floating point number
    dim = math.sqrt(n)

    # The following for loop creates n nodes and place them randomly in the 2D area.
    # If one want to consider a different placement, e.g., for modeling communities, one only need to change this part.
    for i in range(n):
        x=random.random()
        y=random.random()
        nodes[i]=(x*dim,y*dim)
        prob[i]=dict()

    for i in tqdm(range(n)):
        # Strong-ties
        for j in range(i+1,n):
            # we add edge only towards next nodes,
            # since edge to previous nodes have been added when these nodes have been processed
            dist = math.sqrt((nodes[i][0]-nodes[j][0])**2 + (nodes[i][1]-nodes[j][1])**2) #Euclidean Distance
            prob[i][j] = 1/(dist**q)
            prob[j][i] = prob[i][j]
            if dist <= r:
                G.add_edge(str(i), str(j))

        # Terms 1/dist(u,v)**q are not probabilities since their sum can be different from 1.
        # To translate them in probabilities we normalize them, i.e, we divide each of them for their sum
        norm=sum(prob[i].values())
        # Weak ties
        for h in range(k):
            # They are not exactly h, since the random choice can return a node s such that edge (i, s) already exists
            # Next instruction allows to choice from the list given as first argument according to the probability distribution given as second argument
            s=numpy.random.choice([x for x in range(n) if x != i],p=[prob[i][x]/norm for x in range(n) if x != i])
            G.add_edge(str(i), str(s))

    return G




if __name__ == '__main__':
    # Test e Analisi delle tempistiche

    parser = ArgumentParser()
    parser.add_argument('--n_jobs', help='numero di processi', type=int, default=4)
    parser.add_argument('--file_name', help='noem file di testo su cui salvare i risultati', type=str, default="diameter.txt")
    
    args = parser.parse_args()

    n_jobs = args.n_jobs
    file_name = args.file_name
    
    G1 = create_graph_from_csv('../data/musae_facebook_edges.csv')
    G2 = create_graph_from_txt('../data/Cit-HepTh.txt', sep='\t', directed=True)

    #df = pd.DataFrame()
    dict_G1 = {'Algorithm':[], 'Time (s)':[], 'Diameter':[]}
    dict_G2 = {'Algorithm':[], 'Time (s)':[], 'Diameter':[]}
    
   
    ##################################################################################
    #################################### Facebook ####################################    
    ##################################################################################
    
    s = f'Facebook\n\ndirected: {G1.is_directed()}, node: {G1.number_of_nodes()}, edges: {G1.number_of_edges()}'
    print(s)
    
    ################################### STANDARD ###################################
    tic = time.time()
    diam = diameter(G1)
    toc = time.time()
    exe_time = round(toc - tic, 3)

    dict_G1["Algorithm"].append("Standard")
    dict_G1["Time (s)"].append(exe_time)
    dict_G1["Diameter"].append(diam)
    t = f'Standard algo\ntime: {exe_time}s, diameter: {diam}'
    print(t)

    ################################### parallel STANDARD ###################################
    tic = time.time()
    diam = parallel_diam(G1, n_jobs)
    toc = time.time()
    exe_time = round(toc - tic, 3)

    dict_G1["Algorithm"].append("Parallel standard")
    dict_G1["Time (s)"].append(exe_time)
    dict_G1["Diameter"].append(diam)
    t = f'Parallel standard algo\ntime: {exe_time}s, diameter: {diam}'
    print(t)

    ################################### OPTIMIZED ###################################
    tic = time.time()
    diam = stream_diam(G1)
    toc = time.time()
    exe_time = round(toc - tic, 3)

    
    dict_G1["Algorithm"].append("Optimized")
    dict_G1["Time (s)"].append(exe_time)
    dict_G1["Diameter"].append(diam)
    t = f'Optimized algo\ntime: {exe_time}s, diameter: {diam}'
    print(t)

    ################################### NETWORKX ###################################
    tic = time.time()
    diam = nx.diameter(G1)
    toc = time.time()
    exe_time = round(toc - tic, 3)

    dict_G1["Algorithm"].append("Networkx")
    dict_G1["Time (s)"].append(exe_time)
    dict_G1["Diameter"].append(diam)
    t = f'Networkx algo\ntime: {exe_time}s, diameter: {diam}'
    print(t)

    df1 = pd.DataFrame(dict_G1, columns=['Algorithm', 'Time (s)', 'Diameter'])
    print(df1)
    df1.to_csv("diameter_musae_facebook_edges.csv", index=False)
    
    ##################################################################################
    #################################### Citation ####################################
    ##################################################################################

    s = f'\nCitation Network\n\ndirected: {G2.is_directed()}, node: {G2.number_of_nodes()}, edges: {G2.number_of_edges()}'
    print(s)
    
    ################################### STANDARD ###################################
    tic = time.time()
    diam = diameter(G2)
    toc = time.time()
    exe_time = round(toc - tic, 3)

    dict_G2["Algorithm"].append("Standard")
    dict_G2["Time (s)"].append(exe_time)
    dict_G2["Diameter"].append(diam)
    t = f'Standard algo\ntime: {exe_time}s, diameter: {diam}'
    print(t)

    ################################### parallel STANDARD ###################################
    tic = time.time()
    diam = parallel_diam(G2, n_jobs)
    toc = time.time()
    exe_time = round(toc - tic, 3)

    dict_G2["Algorithm"].append("Parallel standard")
    dict_G2["Time (s)"].append(exe_time)
    dict_G2["Diameter"].append(diam)
    t = f'Parallel standard algo\ntime: {exe_time}s, diameter: {diam}'
    print(t)

    ################################### OPTIMIZED ###################################
    tic = time.time()
    diam = stream_diam(G2)
    toc = time.time()
    exe_time = round(toc - tic, 3)

    dict_G2["Algorithm"].append("Optimized")
    dict_G2["Time (s)"].append(exe_time)
    dict_G2["Diameter"].append(diam)
    t = f'Optimized algo\ntime: {exe_time}s, diameter: {diam}'
    print(t)

    ################################### NETWORKX ###################################
    ### Solo per connessi

    df2 = pd.DataFrame(dict_G2, columns=['Algorithm', 'Time (s)', 'Diameter'])
    df2.to_csv("diameter_Cit-HepTh.csv", index=False)
