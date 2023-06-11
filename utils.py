from collections import deque
import csv
import random

import networkx as nx
from pathlib import Path
from queue import PriorityQueue

from lesson5 import GenWS2DG

def auction_results(allocations: dict, bids: dict, payments: dict):
    rw = sum(payments.values())
    sw = 0.
    for bidder, alloc in allocations.items():
        if alloc:
            sw += bids[bidder]
    
    return sw, rw

def create_auction(num_nodes, r = 2.71,k = 1, q=4):
    G = GenWS2DG(num_nodes,r,k,q)
    # scelta casuale del seller
    seller = random.choice(list(G.nodes()))
    # costruzione di seller_net
    seller_net = {bidder for bidder in G[seller]}

    # costruzione dizionario reports
    reports = {}
    level = deque([seller])
    visited = [seller]

    while len(level) > 0:

        n = level.popleft()
        for c in G[n]:
            if c not in visited:
                level.append(c)
                visited.append(c)
                if n not in reports:
                    reports[n] = [c]
                else:
                    reports[n].append(c)

    del reports[seller]
    # costruzione di bids
    bids = {}
    for n in G.nodes():
        bid = random.randrange(1, 10000, 1)
        bids[n] = bid

    return seller_net, reports, bids

def BFS(G,u):
    """
    A BFS algorithm that returns the set of nodes reachable from u in the graph G

    Parameters
    ----------
    G: nx.Graph or nx.DiGraph
        A networkx undirected or directed graphs
    u: node
        A node of G

    Returns
    ---------
    set:
        the set of nodes reachable from u in the graph G
    """
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
    return visited


#Returns the top k nodes of G according to the centrality measure "measure"
def top(G,cen,k):
    pq = PriorityQueue()
    for u in G.nodes():
        x = -cen[u]
        pq.put((x,u))  # We use negative value because PriorityQueue returns first values whose priority value is lower
    
    out={}
    for i in range(k):
        x = pq.get()
        out[x[1]] = -x[0]
    return out

#Returns the top k nodes of G according to the centrality measure "measure"
def bottom(G,cen,k):
    pq = PriorityQueue()
    for u in G.nodes():
        x = cen[u]
        pq.put((x,u))  # We use negative value because PriorityQueue returns first values whose priority value is lower
    
    out={}
    for i in range(k):
        x = pq.get()
        out[x[1]] = -x[0]
    return out


def create_graph_from_csv(filename, sep=',', directed=False):
    path = Path(filename)
    if not path.is_file():
        raise ValueError('File Not Exist!')

    if not directed:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=sep)
        # l'intestazione del csv non viene considerata
        header = next(reader)
        #header =1
        if header is not None:
            for row in reader:
                if len(row) == 2:
                    u, v = row
                    G.add_edge(u, v)
                else:
                    u, v, _, _ = row
                    G.add_edge(u, v)

    return G


def create_graph_from_txt(filename, sep=',', directed=False):
    path = Path(filename)
    if not path.is_file():
        raise ValueError('File Not Exist!')

    if not directed:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    file = open(filename, 'r')
    lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line.startswith('#'):
            row = line.split(sep)
            if len(row) == 2:
                u, v = row
                G.add_edge(u, v)

            else:
                raise ValueError('Format File Error!')

    return G


def chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# def chunks(data, size):
#     idata = iter(data)
#     for i in range(0, len(data), size):
#         yield {k: data[k] for k in it.islice(idata, size)}

if __name__ == '__main__':
    G = create_graph_from_csv('data/musae_facebook_edges.csv')
    print('Numero di nodi:', G.number_of_nodes())
    print('Numero di archi: ', G.number_of_edges())
    G2 = create_graph_from_txt('data/Cit-HepTh.txt', directed=True, sep='\t')
    print('Numero di nodi:', G2.number_of_nodes())
    print('Numero di archi: ', G2.number_of_edges())
