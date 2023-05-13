# This is a sample Python script.
import sys
 
# adding Folder_2 to the system path
sys.path.insert(0, '../')

import math

import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from threading import Thread
from joblib import Parallel, delayed
from queue import PriorityQueue
from utils import *

__all__ = ['linear_pageRank', 'pageRank', 'pageRank_threads', 'linear_parallel_ranking']

def linear_pageRank(G: nx.DiGraph, it=80, tol=1e-5, s=0.85):
    """
    Algoritmo di page ranking mediante il calcolo dell'autovettore della "link matrix" Av = v.
    Ogni valore A[i,j] della matrice A rappresenta 1/out_degree(j) ossia il numero di archi uscenti dal nodo j,
    collegato al nodo i-esimo. Se j non è collegato a i il valore A[i,j]=0.
    Il vettore v ha come componenti i punteggi dei singoli nodi del grafo
    :param tol: valore float usato dal criterio di stop per ritenere stabili i risultati dell'algoritmo
    :param G: grafo orientato della libreria networkx
    :param it: intero che rappresenta il numero d'iterazioni
    :return: dizionario contenente i punteggi per ogni nodo
    """
    # NOTA: I punteggi sono aggiornati dopo aver calcolato tutti i nuovi punteggi di ogni vertice, quindi
    # il nuovo punteggio di un nodo non influisce sui nodi vicini all'iterazione corrente ma solo alla successiva

    nodes = [n for n in G.nodes()]

    # inizializzazione del vettore dei punteggi 1/n
    scores = np.ones(G.number_of_nodes()) * 1 / G.number_of_nodes()
    # inizializzazione della link matrix
    A = np.zeros((G.number_of_nodes(), G.number_of_nodes()))

    if G.is_directed():
        degree_func = G.out_degree
    else:
        degree_func = G.degree
    # costruzione della link matrix
    ids = {n: id_n for id_n, n in enumerate(G.nodes())}
    for u in G.nodes():
        id_u = ids[u]
        for v in G[u]:
            id_v = ids[v]
            A[id_v, id_u] = 1. / degree_func(u)

    # calcolo dell'autovettore
    step = 0
    convergence = False
    while step < it and not convergence:
        # prodotto scalare tra la link matrix e i punteggi
        new_scores = np.dot(A, scores) * s
        new_scores += (1 - s) / G.number_of_nodes()
        # mean absolute error
        diff = np.abs(scores - new_scores)
        convergence_value = np.sum(diff)

        # criterio di stop
        if convergence_value < tol:
            convergence = True
        # aggiornamento dei punteggi
        scores = new_scores

        step += 1

    # costruzione del dizionario (nodo,score)
    # map_score = {}
    map_score = {node: score for node, score in zip(nodes, scores)}
    # for i, node in enumerate(nodes):
    #     map_score[node] = scores[i]
    # ordinare i nodi per punteggio
    return map_score


# La complessità dell'algoritmo è (it*O(deg)). Dagli esperimenti si è notato che aumentando il numero d'iterazioni e
# quindi il numero di nodi attraversati dal visitatore (it = nodi_visitati + 1) i risultati dell'algoritmo random
# walker convergono con i risultati ottenuti con il linear ranking.
def random_walker_ranking(G: nx.DiGraph, it=80, alpha=0.9):
    """
    Algoritmo di page ranking con l'uso del random walker.
    Con l'algoritmo del random walker tracciamo cammini casuali sul grafo, tenendo traccia
    del numero di volte che è un nodo è stato attraversato. Questa informazione ci permette di capire
     dove il traffico converge, in altre parole la "facilità" di raggiungibilità di una certo nodo del grafo
    :param G: grafo orientato della libreria networkx
    :param it: intero che rappresenta numero d'iterazioni
    :param alpha: probabilità di assenza del teletrasporto (1 - alpha = probabilità di teletrasporto)
    :return: dizionario contenente i punteggi per ogni nodo
    """
    # dizionario con frequenze di visita per ogni nodo
    map_score = {}
    for n in G.nodes():
        map_score[n] = 0
    # lista contenente i nodi del grafo
    nodes = list(map_score.keys())

    # nodo corrente su cui sosta il visitatore
    c_node = random.choice(nodes)
    # aggiornamento delle visite del nodo di partenza
    map_score[c_node] += 1
    step = 0
    while step < it:
        # definizione dei criteri di scelta del nodo successore

        # teletrasporto, tecnica utilizzata per "sfuggire" dalle spider traps e dai nodi senza archi uscenti
        successors = nodes
        if np.random.rand() < alpha:
            # scelta di un nodo casuale tra i vicini del nodo corrente
            neighbors = [i for i in G.neighbors(c_node)]
            successors = neighbors
            # dead ends (vicolo cieco), caso in cui il visitatore non può raggiungere altri nodi
            if not neighbors:
                successors = nodes

        n_node = random.choice(successors)
        # aggiornamento visite nodo successore
        map_score[n_node] += 1
        c_node = n_node
        step += 1

    return map_score


def pageRank(G, s=0.85, step=100, confidence=0):
    time = 0
    n = nx.number_of_nodes(G)
    done = False

    # At the beginning, I choose the starting node uniformly at the random.
    # Hence, every node has the same probability of being verified at the beginning.
    rank = {i: float(1) / n for i in G.nodes()}

    if G.is_directed():
        degree_func = G.out_degree
    else:
        degree_func = G.degree

    while not done and time < step:
        time += 1

        # tmp contains the new rank with probability 1-s, I restart the random walk. Hence, each node is visited at
        # the next step at least with probability (1-s)*1/n
        tmp = {i: float(1 - s) / n for i in G.nodes()}

        for i in G.nodes():
            for j in G[i]:
                # with probability s, I follow one of the link on the current page.
                # So, if I am on page i with probability rank[i], at the next step I would be on page j at which i links
                # with probability s*rank[i]*probability of following link (i,j) that is 1/out_degree(i)
                tmp[j] += float(rank[i] * s) / degree_func(i)

        # computes the difference between the old rank and the new rank and updates rank to contain the new rank
        diff = 0
        for i in G.nodes():
            # difference is computed in L1 norm.
            # Alternatives are L2 norm (Euclidean Distance) and L_infinity norm (maximum pointwise distance)
            diff += abs(rank[i] - tmp[i])
            rank[i] = tmp[i]

        if diff <= confidence:
            done = True

    return rank


# -------------Versioni parallelizzate


def pageRank_threads(G: nx.DiGraph, it=2, workers=2, s=0.85, tol=1.0e-6):
    """
    Versione parallelizzata dell'algoritmo pageRank
    :param tol:
    :param G:
    :param it:
    :param workers:
    :param s:
    :return:
    """
    # dizionario con i seguenti record (nodo,[in-neighbors])
    partitions = partition_graph(G, workers)
    global_rank = {n: 1 / nx.number_of_nodes(G) for n in G.nodes()}
    threads = []

    if G.is_directed():
        degree_func = G.out_degree
    else:
        degree_func = G.degree

    for i in range(it):
        shared_dict = {n: (1 - s) / nx.number_of_nodes(G) for n in G.nodes()}
        # scatter phase
        for partition in partitions:
            # ogni thread lavora su un sottoinsieme di record del dizionario condiviso
            thread = Thread(target=compute_pageRank, args=(partition, global_rank, shared_dict, s, degree_func))
            threads.append(thread)
            thread.start()

        np_rank = np.array(global_rank.values())
        np_shared = np.array(shared_dict.values())
        conv = np.sum(np.abs(np_rank - np_shared))
        if conv < tol:
            break

        for thread in threads:
            thread.join()
        # Dopo che ogni thread ha calcolo i punteggi dei vertici assegnati viene aggiornato il rank globale
        # gather phase
        global_rank = shared_dict.copy()

    return global_rank


# funzione di supporto dell'algoritmo pageRank_threads
def compute_pageRank(partition, global_rank, shared_dict, s, degree_func):
    for c in partition.keys():
        for n in partition[c]:
            shared_dict[c] += s * global_rank[n] / degree_func(n)


# funzione di supporto dell'algoritmo pageRank_threads
def partition_graph(G, num_worker):
    map = {n: [] for n in G.nodes}

    for n in G.nodes():
        for c in G[n]:
            map[c].append(n)

    return divide_dict(map, num_worker)


# funzione di supporto dell'algoritmo pageRank_threads
def divide_dict(d, n):
    """
    Divide il dizionario d in n dizionari.
    """
    # Calcola la lunghezza del dizionario
    dict_length = len(d)

    # Calcola il numero di elementi in ogni dizionario
    chunk_size = dict_length // n

    # Calcola il numero di dizionari che avranno un elemento in più
    num_larger_chunks = dict_length % n

    # Inizializza una lista vuota per i dizionari risultanti
    result_dicts = []

    # Itera su ogni blocco di elementi
    start_index = 0
    for i in range(n):
        # Calcola la fine dell'indice del blocco corrente
        end_index = start_index + chunk_size

        # Se questo è l'ultimo blocco con un elemento in più, aggiungi un elemento in più
        if i < num_larger_chunks:
            end_index += 1

        # Estrai il blocco di elementi corrente dal dizionario
        current_chunk = {k: v for idx, (k, v) in enumerate(d.items()) if start_index <= idx < end_index}

        # Aggiungi il blocco di elementi alla lista dei dizionari risultanti
        result_dicts.append(current_chunk)

        # Aggiorna l'indice d'inizio per il prossimo blocco
        start_index = end_index

    return result_dicts


# Non utilizzabile su grafi molto grandi
def linear_parallel_ranking(G: nx.DiGraph, workers: int, it=1, s=0.85, tol=1.0e-6):
    """
    Versione parallelizzata dell'algoritmo linear_ranking.

    :param G:
    :param workers:
    :param it:
    :return:
    """

    nodes = list(G.nodes())
    ids = {n: id_n for id_n, n in enumerate(nodes)}
    num_nodes = nx.number_of_nodes(G)
    A = np.zeros((num_nodes, num_nodes))

    if G.is_directed():
        degree_func = G.out_degree
    else:
        degree_func = G.degree

    # costruzione della matrice A
    for n in G.nodes():
        id_n = ids[n]
        for c in G[n]:
            id_c = ids[c]
            A[id_c, id_n] = 1. / degree_func(n)

    # inizializzazione dei punteggi
    scores = np.ones(num_nodes) * 1 / num_nodes
    # numero di blocchi per riga
    num_blocks_per_row = int(math.sqrt(workers))

    for i in range(it):
        # ogni processo effettua un prodotto scalare su una regione della matrice A (blocco) e la corrispondente
        # porzione del vettore dei punteggi
        with Parallel(n_jobs=workers) as parallel:
            partial_rank = parallel(delayed(scalar_product)(A, b, s) for A, b in matrix_partition(A, scores, workers))

        # combinazione dei risultati dei processi

        # versione sequenziale
        new_scores = []
        for i in range(0, len(partial_rank), num_blocks_per_row):
            new_scores.append(np.sum(partial_rank[i:i + num_blocks_per_row], axis=0))

        # versione parallelizzata
        # il numero di processi da utilizzare per combinare i risultati è pari a num_block_per_row
        # with Parallel(n_jobs=num_blocks_per_row) as parallel:
        #     new_scores = parallel(
        #         delayed(np.sum)(rank_chunk, axis=0) for rank_chunk in
        #         partial_rank_partition(partial_rank, num_blocks_per_row))

        new_scores = np.array(new_scores).flatten()
        new_scores += (1 - s) / num_nodes

        # criterio di stop
        diff = np.abs(scores - new_scores)
        convergence_value = np.sum(diff)
        if convergence_value < tol:
            break

        # aggiornamento del ranking
        scores = new_scores.copy()

    # costruzione del dizionario dei punteggi
    scores = {n: scores[i] for i, n in enumerate(nodes)}

    return scores


def matrix_partition(A: np.array, scores: np.array, num_worker: int):
    # numero righe/colonne della matrice A
    size = A.shape[0]
    # dimensione del blocco, ogni blocco ha lo stesso numero di righe e colonne
    block_size = size / math.sqrt(num_worker)

    # nel caso in cui il blocco non può essere una sotto-matrice quadrata viene lanciata un'eccezione
    if size % np.sqrt(num_worker) != 0:
        raise ValueError(f"Non è possibile suddividere equamente i blocchi tra i processi: {block_size}")
    else:
        block_size = int(block_size)

    for i in range(0, size, block_size):
        for j in range(0, size, block_size):
            yield A[i:i + block_size, j:j + block_size], scores[j:j + block_size]


def scalar_product(A: np.array, b: np.array, s: float):
    """
    Prodotto scalare tra la matrice A e b, e il risultato viene moltiplicato per la costante s
    :param A: matrice numpy
    :param b: array numpy
    :param s: costante di tipo float
    :return: array numpy
    """
    return np.dot(A, b) * s


def partial_rank_partition(partial_rank, num_blocks_per_row):
    for i in range(0, len(partial_rank), num_blocks_per_row):
        yield partial_rank[i:i + num_blocks_per_row]


if __name__ == '__main__':
    # G = create_graph_from_txt('../data/Cit-HepTh.txt', directed=True, sep='\t')
    G = nx.DiGraph()
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')
    G.add_edge('A', 'D')
    G.add_edge('B', 'A')
    G.add_edge('B', 'D')
    G.add_edge('D', 'B')
    G.add_edge('D', 'C')
    G.add_edge('C', 'A')
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')
    G.add_edge('B', 'E')
    G.add_edge('B', 'D')
    G.add_edge('E', 'A')
    G.add_edge('E', 'H')
    G.add_edge('H', 'A')
    G.add_edge('D', 'H')
    G.add_edge('D', 'A')
    G.add_edge('F', 'A')
    G.add_edge('C', 'F')
    G.add_edge('C', 'G')
    G.add_edge('G', 'A')

    print("pageRank")
    p1 = pageRank(G)
    print(top(G, p1, 4))
    print("page2")
    p2 = nx.pagerank(G)
    print(top(G, p2, 4))
    print("pageparallel")
    p2 = linear_pageRank(G)
    print(top(G, p2, 4))

""" 
    # collego ad S le pagine più popolari
    # G.add_edge('A', 'S')
    # G.add_edge('B', 'S')
    # G.add_edge('C', 'S')
    # spider traps
    # G.add_edge('F','T')
    # G.add_edge('T','F')
    #
    # auths, hubs = hits_ranking(G, 20)
    # print(np.sum(list(auths.values())))
    # walker = random_walker_ranking(G, 7000, 0.9)
    #
    # fig, ax = plt.subplots(1, 2)
    # fig.canvas.manager.set_window_title('Hits ranking')
    # ax[0].set_title('Authority')
    # ax[0].bar(list(auths.keys()), list(auths.values()), color='red')
    # ax[1].set_title('Hubs')
    # ax[1].bar(list(hubs.keys()), list(hubs.values()))
    #
    linear = linear_ranking(G, 70)
    # # La somma dei punteggi di tutti i nodi è unitaria nel linear ranking
    # # questo mette in evidenza quanto i nodi distribuiscono uniformemente ai propri vicini il loro punteggio
    # # print(np.sum(list(linear.values())))
    # print("Page rank networkx", nx.betweenness_centrality(G))
    # print(linear)
    # # risultati linear ranking
    fig = plt.figure(2)
    ax = fig.add_subplot()
    fig.canvas.manager.set_window_title('Linear ranking')
    ax.bar(list(linear.keys()), list(linear.values()))
    # # risultati random walker
    # fig2 = plt.figure(3)
    # ax2 = fig2.add_subplot()
    # fig2.canvas.manager.set_window_title('Random walker')
    # ax2.bar(list(walker.keys()), list(walker.values()))

    # # creazione di un grafo casuale
    # G = gnp_random_graph(1000, 0.6, directed=True)
    # # acquisizione tempi del linear ranking
    # tic = time.time()
    # linear_ranking(G, 70)
    # toc = time.time()
    # print(f'Linear ranking: {toc - tic}s')
    # # acquisizione tempi del random walker
    # tic = time.time()
    # random_walker_ranking(G, 1000 * 4, 0.9)
    # toc = time.time()
    # print(f'Random walker: {toc - tic}s')

    # plt.show()
    tic = time.time()
    print(pageRank(G, 1))
    toc = time.time()
    print(toc - tic)
"""
