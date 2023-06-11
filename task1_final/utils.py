import csv

import networkx as nx
from pathlib import Path
from task1.triangles import *
from models import *


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


def is_directed_graph(filename, sep):
    file = open(filename, 'r')
    lines = file.readlines()

    for i in tqdm(range(len(lines))):
        line = lines[i]
        line = line.strip()
        edge = line.split(sep)
        u, v = edge
        for j in range(i + 1, len(lines)):
            new_line = lines[j]
            u2, v2 = new_line.split(sep)
            if v2 == u:
                return True
    return False


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
                if not G.has_edge(u, v):
                    G.add_edge(u, v)

            else:
                raise ValueError('Format File Error!')

    return G


def mean(array):
    m = 0.
    arr_norm = array / np.sum(array)
    for i in range(len(arr_norm)):
        m += i * arr_norm[i]
    return m


def standard_deviation(array):
    variance = 0.

    m = mean(array)
    arr_norm = array / np.sum(array)
    for i in range(len(arr_norm)):
        variance += arr_norm[i] * (i - m) ** 2
    return math.sqrt(variance)


def save_model(g: nx.Graph, path):
    with open(path, "a") as model_file:
        for edge in g.edges():
            edge_str = str(edge[0]) + " " + str(edge[1]) + "\n"
            model_file.write(edge_str)


def degree_distribution_func(G: nx.Graph):
    """
    Calcola le occorrenze in un grafo di ogni grado
    :param G: grafo della libreria networkx
    :return: restituisce un array numpy contenenti le occorrenze dei nodi per ogni grado, dal grado zero al massimo grado
    """
    if G.is_directed():
        degree_func = G.in_degree
    else:
        degree_func = G.degree

    # calcolo del massimo grado
    max_in_deg = 0
    for node in G.nodes():
        degree = degree_func(node)
        if degree > max_in_deg:
            max_in_deg = degree

    power_law_func = np.zeros(max_in_deg + 1)

    for node in G.nodes():
        in_deg = degree_func(node)
        power_law_func[in_deg] += 1
    return power_law_func


def measures(g: nx.Graph):
    meas = {}
    tr = triangles_nodeIteratorN(g)
    num_edges = g.number_of_edges()
    num_nodes = g.number_of_nodes()
    num_components = nx.number_connected_components(g)
    if num_components == 1:
        diameter = nx.approximation.diameter(g)
    else:
        diameter = 'inf'
    communities = len(nx.community.louvain_communities(g))
    coeff = nx.average_clustering(g)

    meas['components'] = num_components
    meas['triangles'] = tr
    meas['edges'] = num_edges
    meas['nodes'] = num_nodes
    meas['diameter'] = diameter
    meas['communities'] = communities
    meas['coeff'] = coeff
    return meas


if __name__ == '__main__':
    net_4 = create_graph_from_txt('net_4', directed=False, sep=' ')
    communities = 0
    steps = 100
    for i in range(steps):
        communities += len(nx.community.louvain_communities(net_4))
    print(communities / steps)
