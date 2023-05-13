import csv

import networkx as nx
from pathlib import Path
from queue import PriorityQueue



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
        if header is not None:
            for row in reader:
                if len(row) == 2:
                    u, v = row
                    G.add_edge(u, v)
                else:
                    raise ValueError('Format File Error!')

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
