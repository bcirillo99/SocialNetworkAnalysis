import numpy as np
import networkx as nx


def linear_hits(G: nx.DiGraph, it=80, wa=0.5, wh=0.5):
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

    return {i: wa*auths_map[i]+wh*hubs_map[i] for i in G.nodes()},auths_map, hubs_map

# HITS
# G is assumed to be a directed graph
# steps and confidence are as for pageRank
# wa and wh are the weights that we assign to the authority value and the hub value of a node, in order to evaluate its rank
def hits(G, step=75, confidence=0, wa=0.5, wh=0.5):
    time = 0
    n = nx.number_of_nodes(G)
    done = False

    hub = {i: float(1) / n for i in G.nodes()} #This contains the hub rank of each node.
    auth = {i: float(1) / n for i in G.nodes()} #This contains the authority rank of each node.

    while not done and time < step:
        time += 1

        htmp = {i:0 for i in G.nodes()} #it contains the new hub rank
        atmp = {i: 0 for i in G.nodes()} #it contains the new authority rank

        atot=0
        for i in G.nodes():
            for e in G.in_edges(i):
                # The authority level increases as better hubs are pointing to him
                atmp[i] += hub[e[0]] #the authority value of a node is the sum over all nodes pointing to him of their hubbiness value
                atot += hub[e[0]] #computes the sum of atmp[i] over all i. It serves only for normalization (each rank is done so that all values always sum to 1)

        htot=0
        for i in G.nodes():
            for e in G.out_edges(i):
                # The hubbiness level increases as it points to better authorities
                htmp[i] += auth[e[1]] #the hubbiness value of a node is the sum over all nodes at which it points of their authority value
                htot += auth[e[1]] #computes the sum of htmp[i] over all i. It serves only for normalization (each rank is done so that all values always sum to 1)

        adiff = 0
        hdiff = 0
        for i in G.nodes():
            adiff += abs(auth[i]-atmp[i] / atot)
            auth[i] = atmp[i] / atot
            hdiff += abs(hub[i] - htmp[i] / htot)
            hub[i] = htmp[i] / htot

        if adiff <= confidence and hdiff <=confidence:
            done = True

    return {i: wa*auth[i]+wh*hub[i] for i in G.nodes()},auth,hub