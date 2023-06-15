# This is a sample Python script.
import sys
# adding Folder_2 to the system path
sys.path.insert(0, '../')
import time
from utils import *
from argparse import ArgumentParser
from shapley import *
from page_rank import *
from voterank import *
from hits import *
import networkx as nx
import random
from scipy.sparse import linalg

#CENTRALITY MEASURES


#The measure associated to each node is exactly its degree
def degree(G):
    cen=dict()
    n = len(G.nodes())
    for i in G.nodes():
        cen[i] = G.degree(i)/(n-1)
    return cen

#The measure associated to each node is the sum of the (shortest) distances of this node from each remaining node
#It is not exavtly the closeness measure, but it returns the same ranking on vertices
def closeness(G):
    cen=dict()

    for u in G.nodes():
        visited=set()
        visited.add(u)
        queue = [u]
        dist = dict()
        dist[u]  = 0
        while queue != []:
            v = queue.pop(0)
            for w in G[v]:
                if w not in visited:
                    queue.append(w)
                    visited.add(w)
                    dist[w] = dist[v] + 1
        cen[u] = sum(dist.values())
    return cen

# Computes edge and vertex betweenness of the graph in input
def betweenness(G):
    edge_btw={frozenset(e):0 for e in G.edges()}
    node_btw={i:0 for i in G.nodes()}

    for s in G.nodes():
        # Compute the number of shortest paths from s to every other node
        tree = [] #it lists the nodes in the order in which they are visited
        spnum = {i:0 for i in G.nodes()} #it saves the number of shortest paths from s to i
        parents = {i:[] for i in G.nodes()} #it saves the parents of i in each of the shortest paths from s to i
        distance = {i:-1 for i in G.nodes()} #the number of shortest paths starting from s that use the edge e
        eflow = {frozenset(e):0 for e in G.edges()} #the number of shortest paths starting from s that use the edge e
        vflow = {i:1 for i in G.nodes()} #the number of shortest paths starting from s that use the vertex i. It is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

        #BFS
        queue = [s]
        spnum[s] = 1
        distance[s] = 0
        while queue != []:
            c = queue.pop(0)
            tree.append(c)
            for i in G[c]:
                if distance[i] == -1: #if vertex i has not been visited
                    queue.append(i)
                    distance[i] = distance[c] + 1
                if distance[i] == distance[c] + 1: #if we have just found another shortest path from s to i
                    spnum[i] += 1
                    parents[i].append(c)

        # BOTTOM-UP PHASE
        while tree != []:
            c = tree.pop()
            for i in parents[c]:
                eflow[frozenset({c,i})] += vflow[c] * (spnum[i]/spnum[c]) #the number of shortest paths using vertex c is split among the edges towards its parents proportionally to the number of shortest paths that the parents contributes
                vflow[i] += eflow[frozenset({c,i})] #each shortest path that use an edge (i,c) where i is closest to s than c must use also vertex i
                edge_btw[frozenset({c, i})] += eflow[frozenset({c,i})] #betweenness of an edge is the sum over all s of the number of shortest paths from s to other nodes using that edge
            if c != s:
                node_btw[c] += vflow[c]  #betweenness of a vertex is the sum over all s of the number of shortest paths from s to other nodes using that vertex
    return edge_btw,node_btw
#The algorithm is quite time-consuming. Indeed, its computational complexity is O(nm).
# optimizations: parallelization, considering only a sample of starting nodes
#The measure associated to each node is its betweenness value
def btw(G):
    return betweenness(G)[1]


if __name__ == '__main__':
    # Test e Analisi delle tempistiche

    parser = ArgumentParser()
    parser.add_argument('--k', help='numero di nodi da far ritornare alla funzione top', type=int, default=5)
    parser.add_argument('--file_name', help='noem file di testo su cui salvare i risultati', type=str, default="centrality.txt")
    
    args = parser.parse_args()

    k = args.k
    file_name = args.file_name

    G1 = create_graph_from_csv('../data/musae_facebook_edges.csv')
    G2 = create_graph_from_txt('../data/Cit-HepTh.txt', sep='\t', directed=True)

    with open(file_name,"w") as fp:
        ##################################################################################
        #################################### Facebook ####################################
        ##################################################################################   
            
        """s = f'Facebook\n\ndirected: {G1.is_directed()}, node: {G1.number_of_nodes()}, edges: {G1.number_of_edges()}'
        fp.write(s+"\n\n")
        print(s)
        
        ################################### degree ###################################
        tic = time.time()
        cen = degree(G1)
        toc = time.time()

        t = f'Degree centrality\ntime: {toc - tic}s, top {k} nodes: {top(G1,cen,k)}'
        fp.write(t+"\n\n")
        print(t)
        #print(nx.degree_centrality(G1))

        ################################### closeness ###################################
        tic = time.time()
        cen = closeness(G1)
        toc = time.time()

        t = f'closeness centrality\ntime: {toc - tic}s, top {k} nodes: {top(G1,cen,k)}'
        fp.write(t+"\n\n")
        print(t)
        #print(nx.closeness_centrality(G1))

        ################################### betweenness ###################################
        tic = time.time()
        cen = btw(G1)
        toc = time.time()

        t = f'betweenness centrality\ntime: {toc - tic}s, top {k} nodes: {top(G1,cen,k)}'
        fp.write(t+"\n\n")
        print(t)
        #print(nx.betweenness_centrality(G1))

        ################################### PageRank ###################################
        tic = time.time()
        cen = pageRank(G1)
        toc = time.time()

        t = f'PageRank centrality\ntime: {toc - tic}s, top {k} nodes: {top(G1,cen,k)}'
        fp.write(t+"\n\n")
        print(t)
        #print(nx.pagerank(G1))

        ################################### Linear PageRank ###################################
        tic = time.time()
        cen = linear_pageRank(G1)
        toc = time.time()

        t = f'Linear PageRank centrality\ntime: {toc - tic}s, top {k} nodes: {top(G1,cen,k)}'
        fp.write(t+"\n\n")
        print(t)
        #print(nx.pagerank(G1))

        ################################### VoteRank ###################################
        tic = time.time()
        cen = voterank(G1)
        toc = time.time()

        t = f'VoteRank centrality\ntime: {toc - tic}s, top {k} nodes: {top(G1,cen,k)}'
        fp.write(t+"\n\n")
        print(t)
        #print(nx.voterank(G1))

        ################################### shapley degree ###################################
        tic = time.time()
        cen = shapley_degree(G1)
        toc = time.time()

        t = f'shapley degree centrality\ntime: {toc - tic}s, top {k} nodes: {top(G1,cen,k)}'
        fp.write(t+"\n\n")
        print(t)

        ################################### shapley threshold ###################################
        tic = time.time()
        cen = shapley_threshold(G1)
        toc = time.time()

        t = f'shapley threshold centrality\ntime: {toc - tic}s, top {k} nodes: {top(G1,cen,k)}'
        fp.write(t+"\n\n")
        print(t)

        ################################### shapley closeness ###################################
        tic = time.time()
        cen = shapley_closeness(G1,positive_decr_fun)
        toc = time.time()

        t = f'shapley closeness centrality\ntime: {toc - tic}s, top {k} nodes: {top(G1,cen,k)}'
        fp.write(t+"\n\n")
        print(t)

        ##################################################################################
        #################################### Citation ####################################
        ##################################################################################

        s = f'\nCitation Network\n\ndirected: {G2.is_directed()}, node: {G2.number_of_nodes()}, edges: {G2.number_of_edges()}'
        fp.write(s+"\n\n")
        print(s)
        ################################### degree ###################################
        tic = time.time()
        cen = degree(G2)
        toc = time.time()

        t = f'Degree centrality\ntime: {toc - tic}s, top {k} nodes: {top(G2,cen,k)}'
        fp.write(t+"\n\n")
        print(t)
        #print(nx.degree_centrality(G2))

        ################################### closeness ###################################
        tic = time.time()
        cen = closeness(G2)
        toc = time.time()

        t = f'closeness centrality\ntime: {toc - tic}s, top {k} nodes: {top(G2,cen,k)}'
        fp.write(t+"\n\n")
        print(t)

        ################################### betweenness ###################################
        tic = time.time()
        cen = btw(G2)
        toc = time.time()

        t = f'betweenness centrality\ntime: {toc - tic}s, top {k} nodes: {top(G2,cen,k)}'
        fp.write(t+"\n\n")
        print(t)"""



        ################################### PageRank ###################################
        tic = time.time()
        cen = pageRank(G2)
        toc = time.time()

        t = f'PageRank centrality\ntime: {toc - tic}s, top {k} nodes: {top(G2,cen,k)}'
        fp.write(t+"\n\n")
        print(t)
        #print(nx.pagerank(G1))

        ################################### Linear PageRank ###################################
        tic = time.time()
        cen = linear_pageRank(G2)
        toc = time.time()

        t = f'Linear PageRank centrality\ntime: {toc - tic}s, top {k} nodes: {top(G2,cen,k)}'
        fp.write(t+"\n\n")
        print(t)

        ################################### VoteRank ###################################
        """tic = time.time()
        cen = voterank(G2)
        toc = time.time()

        t = f'VoteRank centrality\ntime: {toc - tic}s, top {k} nodes: {top(G1,cen,k)}'
        fp.write(t+"\n\n")
        print(t)"""
        #print(nx.voterank(G2))

        ################################### HITS ###################################
        tic = time.time()
        cen_both, cen_auth, cen_hubs = hits(G2)
        toc = time.time()

        t = f'HITS-both centrality\ntime: {toc - tic}s, top {k} nodes: {top(G2,cen_both,k)}'
        t1 = f'HITS-authority centrality\ntime: {toc - tic}s, top {k} nodes: {top(G2,cen_auth,k)}'
        t2 = f'HITS-hubiness centrality\ntime: {toc - tic}s, top {k} nodes: {top(G2,cen_hubs,k)}'
        fp.write(t+"\n"+t1+"\n"+t2+"\n\n")
        print(t)
        print(t1)
        print(t2)

        ################################### linear HITS ###################################
        tic = time.time()
        cen_both, cen_auth, cen_hubs = linear_hits(G2)
        toc = time.time()

        t = f'linear HITS-both centrality\ntime: {toc - tic}s, top {k} nodes: {top(G2,cen_both,k)}'
        t1 = f'linear HITS-authority centrality\ntime: {toc - tic}s, top {k} nodes: {top(G2,cen_auth,k)}'
        t2 = f'linear HITS-hubiness centrality\ntime: {toc - tic}s, top {k} nodes: {top(G2,cen_hubs,k)}'
        fp.write(t+"\n"+t1+"\n"+t2+"\n\n")
        print(t)
        print(t1)
        print(t2)

        h, a = nx.hits(G2)
        t1 = f'Ha centrality\ntime: {toc - tic}s, top {k} nodes: {top(G2,a,k)}'
        t2 = f'Hh centrality\ntime: {toc - tic}s, top {k} nodes: {top(G2,h,k)}'
        #fp.write(t+"\n"+t1+"\n"+t2+"\n\n")
        #print(t)
        print(t1)
        print(t2)


        ################################### shapley degree ###################################
        tic = time.time()
        cen = shapley_degree(G2)
        toc = time.time()

        t = f'shapley degree centrality\ntime: {toc - tic}s, top {k} nodes: {top(G2,cen,k)}'
        fp.write(t+"\n\n")
        print(t)

        ################################### shapley threshold ###################################
        tic = time.time()
        cen = shapley_threshold(G2)
        toc = time.time()

        t = f'shapley threshold centrality\ntime: {toc - tic}s, top {k} nodes: {top(G2,cen,k)}'
        fp.write(t+"\n\n")
        print(t)

        ################################### shapley closeness ###################################
        tic = time.time()
        cen = shapley_closeness(G2,positive_decr_fun)
        toc = time.time()

        t = f'shapley closeness centrality\ntime: {toc - tic}s, top {k} nodes: {top(G2,cen,k)}'
        fp.write(t+"\n\n")
        print(t)


