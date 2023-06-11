# This is a sample Python script.
import collections

import networkx as nx
from joblib import Parallel, delayed
import math
import numpy as np
import time


# from utils import *


def triangles(G: nx.Graph):
    """
    Algoritmo standard per il conteggio dei triangoli in un grafo indiretto o diretto
    :param G: grafo della libreria networkx
    :return: numero di triangoli di un grafo indiretto
    """
    triangles = 0

    if G.is_directed():
        scalar_factor = 1 / 3
    else:
        scalar_factor = 1 / 6

    for u in G.nodes():
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


def parallel_triangles(G: nx.Graph, n_jobs: int):
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


if __name__ == '__main__':
    # Test e Analisi delle tempistiche

    # grafo indiretto
    # G = create_graph_from_csv('../data/musae_facebook_edges.csv')
    # G = nx.complete_graph(50)
    # print(eigen_triangles(G))
    G = nx.Graph()
    G.add_edge(2, 1)
    G.add_edge(1, 3)
    G.add_edge(3, 2)
    G.add_edge(3, 4)
    G.add_edge(2, 4)
    G.add_edge(2, 5)
    G.add_edge(2, 5)
    G.add_edge(5, 6)
    G.add_edge(6, 4)
    G.add_edge(4, 4)
    G.add_edge(4, 5)
    G.add_edge(4, 8)
    G.add_edge(9, 10)
    G.add_edge(9, 11)
    G.add_edge(10, 11)
    G.add_edge(11, 3)

    # print(eigen_triangles(G))
    print("standard", triangles(G))
    # tic = time.time()
    counter_net = sum(nx.triangles(G).values()) / 3
    print(counter_net)
    # toc = time.time()
    # print(f'networkx: {toc - tic}s, counter: {counter_net}')
    # tic = time.time()
    # counter = triangles_nodeIteratorN(G)
    # toc = time.time()
    #
    # print(f'nodeIteratorN: {toc - tic}s, counter: {counter}, error: {counter - counter_net}')
    # tic = time.time()
    # counter = triangles(G)
    # # print(triangles_list)
    # toc = time.time()
    # print(f'algoritmo standard: {toc - tic}s, counter: {counter}, error: {counter - counter_net}')
    # tic = time.time()
    # counter = triangles_nodeIteratorPlusPlus(G)
    # toc = time.time()
    # print(f'nodeIterator++: {toc - tic}s, counter: {counter}, error: {counter - counter_net}')
    # tic = time.time()
    # counter = parallel_triangles(G, 4)
    # toc = time.time()
    # print(f'nodeIteratorNParallelo: {toc - tic}s, counter: {counter}')

    # grafo diretto
    #
    # G2 = nx.DiGraph()
    #
    G2 = create_graph_from_txt('../data/Cit-HepTh.txt', directed=True, sep='\t')
    tic = time.time()
    counter = triangles(G2)
    toc = time.time()

    print(triangles(G2))
    # print(f'time: {toc - tic}s, counter: {counter}')
    # tic = time.time()
    # print("counter dfs", dfs_triangles(G2))
    # print("eigen", eigen_triangles(G2))
    # toc = time.time()
    # print(toc - tic)
