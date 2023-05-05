import numpy as np
import networkx as nx


def hits_ranking(G: nx.DiGraph, it=80):
    # inizializzazione
    auths = np.ones(G.number_of_nodes())
    hubs = np.ones(G.number_of_nodes())

    nodes = sorted(list(G.nodes()))
    # matrice di adiacenza usata per il calcolo dei punteggi di hub
    adj_mat = nx.adjacency_matrix(G, nodes).todense()
    # trasposta della matrice di adiacenza usata per il calcolo dei punteggi di authority
    adj_mat_t = np.transpose(adj_mat)
    step = 0
    while step < it:
        # calcolo dei nuovi punteggi di authority
        auths_new = np.dot(adj_mat_t, hubs)
        # calcolo dei nuovi punteggi di hub
        hubs_new = np.dot(adj_mat, auths)
        # normalizzazione
        auths = auths_new / np.sum(auths_new)
        hubs = hubs_new / np.sum(hubs_new)
        step += 1
    # costruzione dei dizionari (node,auth_score) e (node,hub_score)
    auths_map = {}
    hubs_map = {}
    for i, n in enumerate(nodes):
        auths_map[n] = auths[i]
        hubs_map[n] = hubs[i]

    return auths_map, hubs_map
