import networkx as nx
import itertools
from heapq import heappush, heappop
import random
import time
import pandas as pd
from scipy.sparse import linalg
from utils import create_graph_from_txt, create_graph_from_csv
from queue import PriorityQueue


REMOVED = '<removed-task>'  # placeholder for a removed task

class PriorityQueue:

    def __init__(self):
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.counter = itertools.count()  # unique sequence count

    def add(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = REMOVED
        return entry[0]

    def pop(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')
    

def hierarchical(G):
     # Create a priority queue with each pair of nodes indexed by distance
    pq = PriorityQueue()
    for u in G.nodes():
        for v in G.nodes():
            if u != v:
                if (u, v) in G.edges() or (v, u) in G.edges():
                    pq.add(frozenset([frozenset(u), frozenset(v)]), 0)
                else:
                    pq.add(frozenset([frozenset(u), frozenset(v)]), 1)
    #
    # Start with a cluster for each node
    clusters = set(frozenset(u) for u in G.nodes())
    #
    done = False
    while not done:
        # Merge closest clusters
        s = list(pq.pop())
        clusters.remove(s[0])
        clusters.remove(s[1])
    #
        # Update the distance of other clusters from the merged cluster
        for w in clusters:
            e1 = pq.remove(frozenset([s[0], w]))
            e2 = pq.remove(frozenset([s[1], w]))
            if e1 == 0 or e2 == 0:
                pq.add(frozenset([s[0] | s[1], w]), 0)
            else:
                pq.add(frozenset([s[0] | s[1], w]), 1)
    #
        clusters.add(s[0] | s[1])
    #
        print(clusters)
        a = input("Do you want to continue? (y/n) ")
        if a == "n":
            done = True


def two_means(G):
    n = G.number_of_nodes()
    # Choose two clusters represented by vertices that are not neighbors
    u = random.choice(list(G.nodes()))
    v = random.choice(list(nx.non_neighbors(G, u)))
    cluster0 = {u}
    cluster1 = {v}
    added = 2

    while added < n:
        # Choose a node that is not yet in a cluster and add it to the closest cluster
        x = random.choice([el for el in G.nodes() if el not in cluster0 | cluster1 and (len(
            set(G.neighbors(el)).intersection(cluster0)) != 0 or len(
            set(G.neighbors(el)).intersection(cluster1)) != 0)])
        if len(set(G.neighbors(x)).intersection(cluster0)) != 0:
            cluster0.add(x)
            added += 1
        elif len(set(G.neighbors(x)).intersection(cluster1)) != 0:
            cluster1.add(x)
            added += 1

    print(cluster0, cluster1)


# Computes edge and vertex betweenness of the graph in input
def betweenness(G):
    edge_btw = {frozenset(e): 0 for e in G.edges()}
    node_btw = {i: 0 for i in G.nodes()}

    for s in G.nodes():
        # Compute the number of shortest paths from s to every other node
        tree = []  # it lists the nodes in the order in which they are visited
        spnum = {i: 0 for i in G.nodes()}  # it saves the number of shortest paths from s to i
        parents = {i: [] for i in G.nodes()}  # it saves the parents of i in each of the shortest paths from s to i
        distance = {i: -1 for i in G.nodes()}  # the number of shortest paths starting from s that use the edge e
        eflow = {frozenset(e): 0 for e in G.edges()}  # the number of shortest paths starting from s that use the edge e
        vflow = {i: 1 for i in
                 G.nodes()}  # the number of shortest paths starting from s that use the vertex i. It is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

        # BFS
        queue = [s]
        spnum[s] = 1
        distance[s] = 0
        while queue != []:
            c = queue.pop(0)
            tree.append(c)
            for i in G[c]:
                if distance[i] == -1:  # if vertex i has not been visited
                    queue.append(i)
                    distance[i] = distance[c] + 1
                if distance[i] == distance[c] + 1:  # if we have just found another shortest path from s to i
                    spnum[i] += spnum[c]
                    parents[i].append(c)

        # BOTTOM-UP PHASE
        while tree != []:
            c = tree.pop()
            for i in parents[c]:
                eflow[frozenset({c, i})] += vflow[c] * (spnum[i] / spnum[
                    c])  # the number of shortest paths using vertex c is split among the edges towards its parents proportionally to the number of shortest paths that the parents contributes
                vflow[i] += eflow[frozenset({c,
                                             i})]  # each shortest path that use an edge (i,c) where i is closest to s than c must use also vertex i
                edge_btw[frozenset({c, i})] += eflow[frozenset({c,
                                                                i})]  # betweenness of an edge is the sum over all s of the number of shortest paths from s to other nodes using that edge
            if c != s:
                node_btw[c] += vflow[
                    c]  # betweenness of a vertex is the sum over all s of the number of shortest paths from s to other nodes using that vertex

    return edge_btw, node_btw


# The algorithm is quite time-consuming. Indeed, its computational complexity is O(nm).
# optimizations: parallelization, considering only a sample of starting nodes

def girman_newman(G):
    # Clusters are computed by iteratively removing edges of largest betweenness
    graph = G.copy()  # We make a copy of the graph. In this way we will modify the copy, but not the original graph
    done = False
    while not done:
        # After each edge removal we will recompute betweenness:
        # indeed, edges with lower betweenness may have increased their importance,
        # since shortest path that previously went through on deleted edges, now may be routed on this new edge;
        # similarly, edges with high betweenness may have decreased their importance,
        # since most of the shortest paths previously going through them disappeared because the graph has been disconnected.
        # However, complexity arising from recomputing betweenness at each iteration is huge.
        # An optimization in this case would be to compute betweenness only once
        # and to remove edges in decreasing order of computed betweenness.
        eb, nb = betweenness(graph)
        # Finding the edge with highest betweenness
        edge = None
        bet = -1
        for i in eb.keys():
            if eb[i] > bet:
                edge = i
                bet = eb[i]
        graph.remove_edge(*list(edge))
        # Deciding whether to stop the clustering procedure
        # To automatize this decision, we can use some measure of performance of the clustering.
        # An example of this measure is the function nx.algorithms.community.partition_quality(G, list(nx.connected_components(graph))).
        # See networx documentation for more details.
        # Given one such measure, one may continue iteration of the algorithm as long as the newly achieved clustering
        # has performance that are not worse than the previous clustering or above a given threshold.
        print(list(nx.connected_components(graph)))
        a = input("Do you want to continue? (y/n) ")
        if a == "n":
            done = True

    return list(nx.connected_components(graph))


def k_means(k, G):
    nodes = list(G.nodes())
    k_nodes = []
    clusters = []
    # nodi aggiunti ai cluster
    added_nodes = []
    # scelta di k nodi casuali
    for _ in range(k):
        done = False
        while not done:
            u = random.choice(nodes)
            # not in [v for v in G[k] for k in k_nodes]
            if u not in k_nodes and u:
                k_nodes.append(u)
                done = True

    # inizializzazione dei cluster
    for n in k_nodes:
        clusters.append({n})
        added_nodes.append(n)

    added = k
    while added < G.number_of_nodes():
        # il nodo x viene scelto casualmente tra nodi che rispettano le seguenti condizioni:
        # 1. x non deve appartenere a nessuno dei k cluster
        # 2. uno dei figli di x deve appartenere a uno dei k cluster
        x = random.choice([el for el in G.nodes() if el not in added_nodes and (len(
            set(G.neighbors(el)).intersection(set(added_nodes))) != 0)])
        for n in range(k):
            if len(set(G.neighbors(x)).intersection(clusters[n])) != 0:
                clusters[n].add(x)
                added_nodes.append(x)
                added += 1
                break
    return clusters


def spectral(G):
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    L = nx.laplacian_matrix(G,nodes).asfptype()

    # Laplacian of a graph is a matrix, with diagonal entries being the degree of the corresponding node and off-diagonal entries being -1 if an edge between the corresponding nodes exists and 0 otherwise
    # print(L) #To see the laplacian of G uncomment this line
    # The following command computes eigenvalues and eigenvectors of the Laplacian matrix.
    # Recall that these are scalar numbers w_1, ..., w_k and vectors v_1, ..., v_k such that Lv_i=w_iv_i.
    # The first output is the array of eigenvalues in increasing order. The second output contains the matrix of eigenvectors: specifically, the eigenvector of the k-th eigenvalue is given by the k-th column of v
    w, v = linalg.eigsh(L, n - 1)
    # print(w) #Print the list of eigenvalues
    # print(v) #Print the matrix of eigenvectors
    # print(v[:,0]) #Print the eigenvector corresponding to the first returned eigenvalue

    # Partition in clusters based on the corresponding eigenvector value being positive or negative
    # This is known to return (an approximation of) the sparset cut of the graph
    # That is, the cut with each of the clusters having many edges, and with few edge among clusters
    # Note that this is not the minimum cut (that only requires few edge among clusters, but it does not require many edge within clusters)
    c1 = set()
    c2 = set()

    for i in range(n):
        print(v[i, 0])
        if v[i, 0] < 0:
            c1.add(nodes[i])
        else:
            c2.add(nodes[i])

    return (c1, c2)


def spectral_k(k, G):
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    L = nx.laplacian_matrix(G,
                            nodes).asfptype()
    w, v = linalg.eigsh(L, n - 1)
    # numero di autovettori da considerare
    n_comp = math.ceil(np.log2(k))
    clusters = {}
    for i in range(n):
        cod = str(np.where(v[i, 0:n_comp] > 0, 1, 0))
        n = nodes[i]
        if cod not in clusters.keys():
            clusters[cod] = {n}
        else:
            clusters[cod].add(n)
    return list(clusters.values())

# How to achieve more than two clusters? 
# (ii) we can use further eigenvectors. For example, we can partition nodes in four clusters by using the first two eigenvectors,
#so that the first (second, respectively) cluster contains those nodes i such that v[i,0] and v[i,1] are both negative (both non-negative, resp.)
# while the third (fourth, respectively) cluster contains those nodes i such that only v[i,0] (only v[i,1], resp.) is negative.




if __name__ == '__main__':
    header = True
    network = "musae_facebook_edges"
    n_jobs = 2

    # G1 = create_graph_from_csv('data/undirected/musae_facebook_edges.csv', directed=False, sep=',')
    G1 = create_graph_from_csv('data/undirected/musae_facebook_edges.csv', sep=',', directed=False)
    nodes = G1.number_of_nodes()
    edges = G1.number_of_edges()
    type_net = G1.is_directed()
    dict_G1 = {'Network': [], 'Directed': [], 'Nodes': [], 'Edges': [], 'Algorithm': [], 'Time (s)': []}
    tic = time.time()
    two_means(G1)
    toc = time.time()
    dict_G1['Network'].append(network)
    dict_G1['Directed'].append(type_net)
    dict_G1['Nodes'].append(nodes)
    dict_G1['Edges'].append(edges)
    dict_G1['Algorithm'].append('two_means')
    dict_G1['Time (s)'].append(toc - tic)
    df1 = pd.DataFrame(dict_G1)
    # print(df1)
    df1.to_csv("clustering_times.csv", index=False, mode='a', header=header)