import networkx as nx
import numpy as np
from collections import deque


def environment(G: nx, s, alive=0.8, seed=None):
    """

    :param G: grafo della libreria networkX diretto o indiretto
    :param s: nodo di un grafo da cui parte la visita
    :param alive: float che definisce la probabilità che un arco sia vivo
    :param seed: intero che rappresenta il seed con il quale si decide per la vita di un edge
    :return: (reward,edges_alive) float che rappresenta la ricompensa e un intero che rappresenta il numero di archi vivi

    """
    num_nodes = G.number_of_nodes()
    # costruzione della matrice che memorizza gli archi visitati
    visited_edges = np.full((num_nodes, num_nodes), False)
    # dizionario contenenti le tuple (indice,nodo)
    id_nodes = {node: i for i, node in enumerate(nodes)}
    reward = 0.
    edges_alive = 0

    if seed is not None:
        np.random.seed(seed)

    # Algoritmo BFS. Termina quando tutti gli archi vivi del grafo raggiungibili
    # dal nodo s sono stati visitati
    level = deque([s])
    while len(level) > 0:

        # nodo correntemente visitato
        n = level.popleft()
        index_n = id_nodes[n]

        for c in G[n]:
            index_c = id_nodes[c]

            if np.random.random() < alive and not visited_edges[index_n, index_c]:
                edge_reward = G.get_edge_data(n, c)['weight']
                reward += edge_reward
                edges_alive += 1
                visited_edges[index_n, index_c] = True
                # nel caso sia un grafo non diretto viene contrassegnato come visitato non solo
                # l'arco (i, j) ma anche l'arco (j, i)
                if not G.is_directed():
                    visited_edges[index_c, index_n] = True
                level.append(c)

    # usando l'algoritmo DFS
    # index_s = id_nodes[s]
    # for c in G[s]:
    #     index_c = id_nodes[c]
    #     if np.random.random() < prob_alive and not visited_edges[index_s, index_c]:
    #         edge_reward = G.get_edge_data(s, c)['weight']
    #         reward += edge_reward
    #         alive += 1
    #         visited_edges[index_s, index_c] = True
    #         # nel caso sia un grafo non diretto viene contrassegnato come visitato non solo
    #         # l'arco (i, j) ma anche l'arco (j, i)
    #         if not directed:
    #             visited_edges[index_c, index_s] = True
    #         environment(G, c, prob_alive, directed, seed, reward, visited_edges)

    return reward, edges_alive


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

G = nx.DiGraph()
prob_alive = 2
G.add_edge('a', 'b', weight=2)
G.add_edge('a', 'c', weight=2)
G.add_edge('c', 'd', weight=2)
G.add_edge('b', 'd', weight=2)

nodes = [n for n in G.nodes()]
print(nodes[0])
print(environment(G, nodes[0], alive=2))
