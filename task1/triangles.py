# This is a sample Python script.
import sys
 
# adding Folder_2 to the system path
sys.path.insert(0, '../')

import collections

import networkx as nx
from joblib import Parallel, delayed
import math
import numpy as np
import time
from utils import *
from argparse import ArgumentParser
import pandas as pd


import itertools as it




#TRIANGLES
#Standard algorithm for a directed or undirected graph
#The problems of this algorithm are two:
#- The same triangle is counted multiple times (six times)
#- For each node, it requires to visit its neighborhood twice (and the neighborhood can be large).
#  For triangles involving only nodes of large degree this cost cannot be avoided even if we count the triangle only once (i.e., we address the first issue).
#  In other words these triangles are a bottleneck for the running time of this algorithm.
#  For the remaining triangles, if u is the node of smaller degree, then the time needed to find them is reasonable.
def triangles(G, nodes=None):
    """
    Standard algorithm for counting triangles in an indirect or direct graph

    Arguments:
        G: graph
    :return: numero di triangoli di un grafo indiretto
    """
    triangles = 0
    if nodes is None:
        nodes = G.nodes()

    if G.is_directed():
        scalar_factor = 1 / 3
    else:
        scalar_factor = 1 / 6

    for u in nodes:
        for v in G[u]:
            for w in G[v]:
                if (u != v) and (v != w) and (w != u) and G.has_edge(w, u):
                    triangles += 1

    return triangles * scalar_factor


# --------------------------------------
def triangles_nodeIteratorPlusPlus(G: nx.Graph):
    """
    Questo algoritmo permette di conteggiare solo una volta ogni triangolo, sfruttando un id assegnato a ogni nodo.
    Un triangolo viene contato solo nel caso in cui id(u)<id(v)<id(w)
    :param G: grafo indiretto della libreria networkx
    :return: numero di triangoli nel grafo G
    """
    if G.is_directed():
        raise ValueError('This algorithm work on undirected graph!')

    # costruzione del dizionario contenente gli id per ogni nodo
    ids = {node: id_n for id_n, node in enumerate(G.nodes())}
    triangles = 0
    for u in G.nodes():
        for v in G[u]:
            if ids[v] > ids[u]:
                for w in G[v]:
                    if ids[w] > ids[v] and G.has_edge(w, u):
                        triangles += 1
    return triangles


def triangles_nodeIteratorN(G: nx.Graph, nodes=None):
    """
    Algoritmo per il conteggio dei triangoli in un grafo indiretto, analizzato nel paper Fast Parallel For Counting
    and Listing Triangles in Big Graphs
    :param G: grafo indiretto della libreria networkx
    :param nodes: lista di nodi, l'algoritmo valuta per ognuno di questi se un triangolo incide su essi
    :return: numero di triangoli nel grafo G
    """
    if G.is_directed():
        raise ValueError('This algorithm work on undirected graph!')

    if nodes is None:
        nodes = G.nodes()
    neighbors = {node: set() for node in G.nodes()}
    # Si potrebbe utilizzare hash(id_n) per contenere la lunghezza dell'id
    ids = {node: id_n for id_n, node in enumerate(G.nodes())}
    triangles = 0

    for u, v in G.edges():
        if ids[u] < ids[v]:
            neighbors[u].add(v)
        if ids[u] > ids[v]:
            neighbors[v].add(u)

    for n in nodes:
        neighbors_n = neighbors[n]
        for v in neighbors_n:
            neighbors_v = neighbors[v]
            # Notiamo che i nodi w in comune tra il vicinato di n e v hanno un id più alto sia di n che di v.
            # Essendo v un nodo vicino ad n id(v) > id(n), quindi id(n)<id(v)<id(w)
            w = neighbors_n.intersection(neighbors_v)
            triangles += len(w)

    return triangles


# -------------------------------------- Algoritmi paralleli
def parallel_nodeIteratorN(G: nx.Graph, n_jobs: int):
    """
    Versione parallela dell'algoritmo nodeIteratorN
    :param G: grafo indiretto della libreria networkx
    :param n_jobs: numero di processi
    :return: numero di triangoli nel grafo G
    """

    def counter_triangles(nodes: list):
        triangles = 0
        for n in nodes:
            neighbors_n = neighbors[n]
            for v in neighbors[n]:
                neighbors_v = neighbors[v]
                w = neighbors_n.intersection(neighbors_v)
                triangles += len(w)
        return triangles

    if G.is_directed():
        raise ValueError('This algorithm work on undirected graph!')

    neighbors = {node: set() for node in G.nodes()}
    ids = {node: id_n for id_n, node in enumerate(G.nodes())}

    for u, v in G.edges():
        if ids[u] < ids[v]:
            neighbors[u].add(v)
        if ids[u] > ids[v]:
            neighbors[v].add(u)

    # ogni processo si occupa di contare i triangoli su un sottoinsieme di vertici
    with Parallel(n_jobs=n_jobs) as parallel:
        result = parallel(delayed(counter_triangles)(chunk) for chunk in
                          chunks(list(G.nodes()), math.ceil(G.number_of_nodes() / n_jobs)))
    return np.sum(result)


def parallel_triangles(G, workers):
    with Parallel(n_jobs=workers) as parallel:
        result = parallel(delayed(triangles)(G, chunk) for chunk in
                          chunks(list(G.nodes()), math.ceil(G.number_of_nodes() / workers)))
    return np.sum(result)


# --------------------------------------


# Nota: Questa funzione calcola il numero triangoli solo per grafi di piccola dimensione.

def eigen_triangles(G: nx):
    """
    La matrice di adiacenza del grafo moltiplicata per se stessa n volte fornisce il numero di cammini di lunghezza n
    da un nodo i a un nodo j. Quindi A^3 fornisce il numero di cammini di lunghezza tre. Un triangolo è un cammino di
    lunghezza tre che parte dal nodo i-esimo e termina nel nodo i-esimo. Di conseguenza la somma della diagonale della matrice di
    adiacenza restituisce il numero di triangoli (Trace(adj_mat)). La traccia di una matrice si può esprimere come
    somma di autovalori della matrice stessa (Trace(adj_mat) = sommatoria(eigen_values)).
    Quindi si perviene al seguente risultato
    triangles(G) = scalar_factor*sommatoria(eigen_values^3) = scalar_factor*Trace(adj_mat^3)
    :param G: grafo diretto o indiretto della libreria networkx
    :return: numero di triangoli nel grafo G
    """

    if G.is_directed():
        scalar_factor = 1 / 3
    else:
        scalar_factor = 1 / 6
    # la matrice di adiacenza ha una crescita spaziale pari a O(n^2), quindi per grafi molto sparsi lo spazio
    # allocato è eccessivo
    adj_mat = nx.adjacency_matrix(G).todense()
    # Il calcolo degli autovalori di una matrice di adiacenza è un operazione molto costosa, si cerca di rimediare a
    # questo collo di bottiglia usando un metodo che approssima il valore degli autovalori, chiamato Lanczos.
    eigen_values, _ = np.linalg.eig(adj_mat)
    triangles = np.sum(eigen_values ** 3)
    return triangles * scalar_factor

#Optimized algorithm
#There are two optimizations.
#
#OPTIMIZATION1: It consider an order among nodes. Specifically, nodes are ordered by degree. In case of nodes with the same degree, nodes are ordered by label.
#In this way a triangle is counted only once. Specifically, from the node with smaller degree to the one with larger degree.
def less(G, edge):
    if G.degree(edge[0]) < G.degree(edge[1]):
        return 0
    if G.degree(edge[0]) == G.degree(edge[1]) and edge[0] < edge[1]:
        return 0
    return 1

#OPTIMIZTION2: It distinguishes between high-degree nodes (called heavy hitters) and low-degree nodes.
#Triangles involving only heavy hitters (that have been recognized to be the bottleneck of the naive algorithm) are handled in a different way respect to remaining triangles.
def num_triangles(G):
    num_triangles = 0
    m = nx.number_of_edges(G)

    # The set of heavy hitters, that is nodes with degree at least sqrt(m)
    # Note: the set contains at most sqrt(m) nodes, since num_heavy_hitters*sqrt(m) must be at most the sum of degrees = 2m
    # Note: the choice of threshold sqrt(m) is the one that minimize the running time of the algorithm.
    # A larger value of the threshold implies a faster processing of triangles containing only heavy hitters, but a slower processing of remaining triangles.
    # A smaller value of the threshold implies the reverse.
    heavy_hitters=set()
    for u in G.nodes():
        if G.degree(u) >= math.sqrt(m):
            heavy_hitters.add(u)

    # Number of triangles among heavy hitters.
    # It considers all possible triples of heavy hitters, and it verifies if it forms a triangle.
    # The running time is then O(sqrt(m)^3) = m*sqrt(m)
    for triple in it.combinations(heavy_hitters,3):
        if G.has_edge(triple[0],triple[1]) and G.has_edge(triple[1], triple[2]) and G.has_edge(triple[2], triple[0]):
            num_triangles+=1

    # Number of remaining triangles.
    # For each edge, if one of the endpoints is not an heavy hitter, verifies if there is a node in its neighborhood that forms a triangle with the other endpoint.
    # This is essentially the naive algorithm optimized to count only ordered triangle in which the first vertex (i.e., u) is not an heavy hitter.
    # Since the size of the neighborhood of a non heavy hitter is at most sqrt(m), the complexity is O(m*sqrt(m))
    for edge in G.edges():
        if edge[0] != edge[1]:
            sel=less(G,edge)
            if edge[sel] not in heavy_hitters:
                for u in G[edge[sel]]:
                    if less(G,[u,edge[1-sel]]) and G.has_edge(u,edge[1-sel]) and u!=edge[1-sel] and u!=G[edge[sel]]:
                        num_triangles +=1

    return num_triangles


if __name__ == '__main__':
    # Test e Analisi delle tempistiche

    parser = ArgumentParser()
    parser.add_argument('--n_jobs', help='numero di processi', type=int, default=4)
    parser.add_argument('--file_name', help='noem file di testo su cui salvare i risultati', type=str, default="triangles.txt")
    
    args = parser.parse_args()

    n_jobs = args.n_jobs
    file_name = args.file_name

    G1 = create_graph_from_csv('../data/musae_facebook_edges.csv')
    G2 = create_graph_from_txt('../data/net_4.txt', sep=' ', directed=False)

    dict_G1 = {'Algorithm':[], 'Time (s)':[], 'Triangles':[]}
    dict_G2 = {'Algorithm':[], 'Time (s)':[], 'Triangles':[]}
    

    ##################################################################################
    #################################### Facebook ####################################  
    ##################################################################################

    s = f'Facebook\n\ndirected: {G1.is_directed()}, node: {G1.number_of_nodes()}, edges: {G1.number_of_edges()}'
    print(s)

    ################################### STANDARD ###################################
    tic = time.time()
    counter = triangles(G1)
    toc = time.time()
    exe_time = round(toc - tic, 3)

    dict_G1["Algorithm"].append("Standard")
    dict_G1["Time (s)"].append(exe_time)
    dict_G1["Triangles"].append(counter)

    t = f'Standard algo\ntime: {exe_time}s, counter: {counter}'
    print(t)

    ################################### parallel STANDARD ###################################
    tic = time.time()
    counter = parallel_triangles(G1, n_jobs)
    toc = time.time()
    exe_time = round(toc - tic, 3)

    dict_G1["Algorithm"].append("Parallel standard")
    dict_G1["Time (s)"].append(exe_time)
    dict_G1["Triangles"].append(counter)

    t = f'Parallel standard algo\ntime: {exe_time}s, counter: {math.ceil(counter)}'
    print(t)

    ################################### OPTIMIZED ###################################
    tic = time.time()
    counter = num_triangles(G1)
    toc = time.time()
    exe_time = round(toc - tic, 3)

    dict_G1["Algorithm"].append("Optimized")
    dict_G1["Time (s)"].append(exe_time)
    dict_G1["Triangles"].append(counter)
    t = f'Optimized algo\ntime: {exe_time}s, counter: {counter}'
    print(t)

    ################################### NETWORKX ###################################
    tic = time.time()
    t = nx.triangles(G1)
    counter=0
    for n in t.values():
        counter+=n
    counter/=3
    toc = time.time()
    exe_time = round(toc - tic, 3)

    dict_G1["Algorithm"].append("Networkx")
    dict_G1["Time (s)"].append(exe_time)
    dict_G1["Triangles"].append(counter)
    t = f'Networkx algo\ntime: {exe_time}s, counter: {counter}'
    print(t)

    ################################### nodeIteratorPlusPlus ###################################
    tic = time.time()
    counter = triangles_nodeIteratorPlusPlus(G1)
    toc = time.time()
    exe_time = round(toc - tic, 3)

    dict_G1["Algorithm"].append("nodeIteratorPlusPlus")
    dict_G1["Time (s)"].append(exe_time)
    dict_G1["Triangles"].append(counter)

    t = f'nodeIteratorPlusPlus algo\ntime: {exe_time}s, counter: {counter}'
    print(t)

    ################################### nodeIteratorN ###################################
    tic = time.time()
    counter = triangles_nodeIteratorN(G1)
    toc = time.time()
    exe_time = round(toc - tic, 3)

    dict_G1["Algorithm"].append("nodeIteratorN")
    dict_G1["Time (s)"].append(exe_time)
    dict_G1["Triangles"].append(counter)

    t = f'nodeIteratorN algo\ntime: {exe_time}s, counter: {counter}'
    print(t)

    ################################### parallel_nodeIteratorN ###################################
    tic = time.time()
    counter = parallel_nodeIteratorN(G1,n_jobs)
    toc = time.time()
    exe_time = round(toc - tic, 3)

    dict_G1["Algorithm"].append("parallel_nodeIteratorN")
    dict_G1["Time (s)"].append(exe_time)
    dict_G1["Triangles"].append(counter)

    t = f'parallel_nodeIteratorN algo\ntime: {exe_time}s, counter: {counter}'
    print(t)
    
    df1 = pd.DataFrame(dict_G1, columns=['Algorithm', 'Time (s)', 'Triangles'])
    print(df1)
    df1.to_csv("triangles_musae_facebook_edges.csv", index=False)
    ##################################################################################
    #################################### Citation ####################################
    ##################################################################################
    
    s = f'\nCitation Network\n\ndirected: {G2.is_directed()}, node: {G2.number_of_nodes()}, edges: {G2.number_of_edges()}'
    print(s)

    ################################### STANDARD ###################################
    tic = time.time()
    counter = triangles(G2)
    toc = time.time()
    exe_time = round(toc - tic, 3)

    dict_G2["Algorithm"].append("Standard")
    dict_G2["Time (s)"].append(exe_time)
    dict_G2["Triangles"].append(counter)
    t = f'Standard algo\ntime: {exe_time}s, counter: {counter}'
    print(t)

    ################################### parallel STANDARD ###################################
    tic = time.time()
    counter = parallel_triangles(G2, n_jobs)
    toc = time.time()
    exe_time = round(toc - tic, 3)

    dict_G2["Algorithm"].append("Parallel standard")
    dict_G2["Time (s)"].append(exe_time)
    dict_G2["Triangles"].append(counter)
    t = f'Parallel standard algo\ntime: {exe_time}s, counter: {counter}'
    print(t)

    ################################### OPTIMIZED ###################################
    tic = time.time()
    counter = num_triangles(G2)
    toc = time.time()
    exe_time = round(toc - tic, 3)

    dict_G2["Algorithm"].append("Optimized")
    dict_G2["Time (s)"].append(exe_time)
    dict_G2["Triangles"].append(counter)

    t = f'Optimized algo\ntime: {exe_time}s, counter: {counter}'
    print(t)

    df2 = pd.DataFrame(dict_G2, columns=['Algorithm', 'Time (s)', 'Triangles'])
    df2.to_csv("triangles_Cit-HepTh.csv", index=False)