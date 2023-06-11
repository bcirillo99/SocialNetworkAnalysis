import matplotlib.pyplot as plt
from utils import *

if __name__ == '__main__':
    # lettura del modello net_4
    net_4 = create_graph_from_txt('net_4', directed=False, sep=' ')
    N = net_4.number_of_nodes()
    # calcolo della distribuzione dei gradi di net_4
    deg_net4 = degree_distribution_func(net_4)
    # calcolo della media della distribuzione del modello net_4
    mean_net4 = mean(deg_net4)
    # calcolo della deviazione standard della distribuzione del modello net_4
    std_net4 = standard_deviation(deg_net4)

    # Per ricreare la distribuzione dei gradi del modello net_4, si è scelto di campionare 20000 campioni da una
    # gaussiana con la stessa media e deviazione standard della distribuzione del modello net_4
    np.random.seed(12)
    degree_list = np.random.normal(mean_net4, std_net4, N).astype(int)

    # istogramma dei campioni estratti
    n, bins, patches = plt.hist(x=degree_list, bins=40, color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Degree list Histogram')
    plt.show()

    # generazione del modello configurationG_result
    G_conf = configurationG(degree_list)
    print('creazione grafo terminata')
    # calcolo delle misure
    # {'components': 1, 'triangles': 2100, 'edges': 221887, 'nodes': 20000, 'diameter': 5, 'communities': 8,
    # 'coeff': 0.0012139030306415875}
    print(measures(G_conf))

    # confronto tra le distribuzioni
    deg_model = degree_distribution_func(G_conf)
    mean_model = mean(deg_model)
    std_model = standard_deviation(deg_model)
    plt.plot(deg_net4, 'b', deg_model, 'g')
    plt.axvline(x=mean_net4, color='b', linestyle='dashed', label='axvline - full height')
    plt.axvline(x=mean_model, color='g', linestyle='dashed', label='axvline - full height')
    plt.title(f'mean net4: {round(mean_net4, 3)}  std net4: {round(std_net4, 3)}\n'
              f'mean confG: {round(mean_model, 3)}  std confG: {round(std_model, 3)}')
    plt.legend(['degree distribution net_4', 'degree distribution confG', 'mean net4', 'mean confG'])
    plt.show()

    # MATCHING:
    # 1. Distribuzione dei gradi

    # CONCLUSIONE
    # Le misure ottenute sono molte diverse da quelle di net_4, escludendo la distribuzione dei gradi
