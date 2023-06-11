import networkx as nx

from utils import *
import matplotlib.pyplot as plt
from models import *

if __name__ == '__main__':
    net_4 = create_graph_from_txt('net_4', directed=False, sep=' ')

    N = net_4.number_of_nodes()
    model = preferentialG(20000, 0.6)
    pl = degree_distribution_func(model)
    plt.plot(pl[1:])
    plt.show()
    # CONCLUSIONE: preferentialG segue un andamento di legge di potenza come si può notare dalla figura,
    # quindi è scartato
