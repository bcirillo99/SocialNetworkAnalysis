import networkx as nx
import numpy as np

from utils import *
from matplotlib import pyplot as plt


def experiment(r, k, q, it=10):
    # r = 2.72
    # r = 2.71
    # k = 1
    # q = 4.1

    m = {'triangles': 0., 'edges': 0., 'diameter': 0., 'components': 0., 'communities': 0., 'coeff': 0., 'mean': 0.,
         'std': 0.}

    for i in range(it):
        g = GenWS2DG(N, r, k, q)
        meas = measures(g)
        m['triangles'] += meas['triangles']
        m['edges'] += meas['edges']
        m['diameter'] += meas['diameter']
        m['components'] += meas['components']
        m['communities'] += meas['communities']
        m['coeff'] += meas['coeff']
        pl = degree_distribution_func(g)
        mean_deg = mean(pl)
        std_deg = standard_deviation(pl)
        m['mean'] += mean_deg
        m['std'] += std_deg
        print('it: ', i, meas, 'mean: ', mean_deg, 'dev: ', std_deg)

    m['triangles'] = m['triangles'] / it
    m['edges'] = m['edges'] / it
    m['diameter'] = m['diameter'] / it
    m['components'] = m['components'] / it
    m['communities'] = m['communities'] / it
    m['coeff'] = m['coeff'] / it
    m['mean'] = m['mean'] / it
    m['std'] = m['std'] / it
    return m


if __name__ == '__main__':
    # Confronto tra le seguenti soluzioni:
    # 1. r = 2.71, k = 1, q = 4
    # 2. r = 2.72, k = 1, q = 4
    N = 20000
    # r = 2.71
    # r = 2.72
    k = 1
    q = 4
    # g = GenWS2DG(N, r, k, q)
    # save_model(g,'ws_models/2.71_4_1.txt')
    # net_4 = create_graph_from_txt('net_4', directed=False, sep=' ')

    # model_1 = create_graph_from_txt('ws_models/1.71_4_1.txt', directed=False, sep=' ')
    # model_2 = create_graph_from_txt('ws_models/2.72_4_1.txt', directed=False, sep=' ')

    # print('Radius: ', nx.radius(model_1), 'Diameter: ', nx.diameter(model_1))
    experiment(2.707, 1, 4)
    # CONCLUSIONE:
    # 1. modello_1 # Raggio: 36 - Diametro 70
    # 2. modello_2 # Raggio: 33 - Diameter:  61
