import matplotlib.pyplot as plt
from utils import *
from models import *

if __name__ == '__main__':
    net_4 = create_graph_from_txt('net_4', directed=False, sep=' ')
    N = net_4.number_of_nodes()
    # calcolo della distribuzione dei gradi di net_4
    deg_net4 = degree_distribution_func(net_4)
    # calcolo della media della distribuzione di net_4
    mean_net4 = mean(deg_net4)
    # calcolo della deviazione standard della distribuzione di net_4
    std_net4 = standard_deviation(deg_net4)

    # Dalla distribuzione dei gradi di net_4, sappiamo che ogni nodo in media ha all'incirca 22 figli, quindi possiamo
    # usare questa informazione per settare il parametro p dell'algoritmo.
    # Ogni nodo i costruisce un arco con un nodo j con probabilità 22/N

    #model = randomG(N, 22.718 / N)
    model = randomG(20000, 40 / N)
    # model = randomG(20000, 60 / N)
    # {'components': 1, 'triangles': 1992, 'edges': 228183, 'nodes': 20000, 'diameter': 5, 'communities': 11,
    # 'coeff': 0.0011336226154432974}
    print(measures(model))

    # calcolo della distribuzione dei gradi del modello generato
    deg_model = degree_distribution_func(model)
    mean_model = mean(deg_model)
    std_model = standard_deviation(deg_model)
    # confronto tra le distribuzioni
    plt.plot(deg_net4, 'b', deg_model, 'g')
    plt.plot(deg_net4, 'b', deg_model, 'g')
    plt.axvline(x=mean_net4, color='b', linestyle='dashed', label='axvline - full height')
    plt.axvline(x=mean_model, color='g', linestyle='dashed', label='axvline - full height')
    plt.title(f'mean net4: {round(mean_net4, 3)}  std net4: {round(std_net4, 3)}\n'
              f'mean randomG: {round(mean_model, 3)}  std randomG: {round(std_model, 3)}')
    plt.legend(['degree distribution net_4', 'degree distribution randomG', 'mean net4', 'mean randomG'])
    plt.show()

    # MATCHING:
    # 1. Distribuzione dei gradi
    # 2. Numero di archi

    # CONCLUSIONE: Aumentando la probabilità p aumenteremo il numero di triangoli ma sposteremo la distribuzione
    # verso destra, quindi non sarà coerente con la distruzione di net_4, e allo stesso tempo ridurremmo il diametro
    # e il numero di comunità in quanto aumentiamo la densità della rete.
