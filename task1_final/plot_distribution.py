from matplotlib import pyplot as plt
import pandas as pd
from utils import *

if __name__ == '__main__':
    k = 1
    q = 4
    net4 = create_graph_from_txt('net_4', directed=False, sep=' ')
    net4_deg = degree_distribution_func(net4)
    net4_mean = mean(net4_deg)
    net4_dev = standard_deviation(net4_deg)
    df = pd.read_csv('ws_csv_results/ws_q_4.csv')
    deg_list = df.loc[:, 'deg_distr']
    rlist = df.loc[:, 'radius']
    mean_dev_list = df.loc[:, 'mean_dev']
    plt.rcParams.update({'font.size': 8})

    for i in range(len(deg_list)):
        deg = str(deg_list[i])
        r = rlist[i]
        deg = deg.replace('[', '')
        deg = deg.replace(']', '')
        deg = deg.replace('\n', '')
        deg = deg.split(' ')

        deg = [float(d) for d in deg if d != '']
        plt.plot(net4_deg, 'b', deg, 'g')
        mean_dev = mean_dev_list[i]
        mean_dev = mean_dev.replace('(', '')
        mean_dev = mean_dev.replace(')', '')
        mean_dev = mean_dev.split(',')
        mean = float(mean_dev[0])
        dev = float(mean_dev[1])
        plt.title(
            f'r = {r}, k = {k}, q = {q}\n mean net4 = {round(net4_mean, 3)}, std net4 = {round(net4_dev, 3)}\n mean '
            f'WS = {round(mean, 3)}, std WS = {round(dev, 3)}')

        plt.axvline(x=net4_mean, color='b', linestyle='dashed', label='axvline - full height')
        plt.axvline(x=mean, color='g', linestyle='dashed', label='axvline - full height')
        plt.legend(['degree distribution net_4', 'degree distribution WS', 'mean net_4', 'mean WS'])
        # plt.show()
        plt.savefig(f'ws_degree_distribution/r_{r}_k_{k}_q_{q}.png')
        plt.clf()
