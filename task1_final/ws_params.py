import pandas as pd
from utils import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    N = 20000
    # Si è scelto di partire da un parametro k molto basso, in quanto il diametro di net_4 è particolarmente alto (68)
    k = 1
    q = 4
    # rlist = [2.6, 2.7, 2.8]
    rlist = [2.7, 2.71, 2.72, 2.73]
    temp_df = {'radius': [], 'nodes': [], 'edges': [], 'diameter': [], 'communities': [], 'components': [],
               'triangles': [], 'coeff': [], 'deg_distr': [], 'mean_dev': []}
    for r in rlist:
        print(f'Costruzione del grafo di raggio {r}')
        g = GenWS2DG(N, r, k, q)
        print('Calcolo delle misurazioni')
        meas_dict = measures(g)
        temp_df['radius'].append(r)
        temp_df['nodes'].append(meas_dict['nodes'])
        temp_df['edges'].append(meas_dict['edges'])
        temp_df['diameter'].append(meas_dict['diameter'])
        temp_df['communities'].append(meas_dict['communities'])
        temp_df['components'].append(meas_dict['components'])
        temp_df['triangles'].append(meas_dict['triangles'])
        temp_df['coeff'].append(meas_dict['coeff'])

        # distribuzione dei gradi
        pl = degree_distribution_func(g)
        temp_df['deg_distr'].append(pl)
        m = round(mean(pl), 3)
        std = round(standard_deviation(pl), 3)
        temp_df['mean_dev'].append((m, std))
        print('radius:', r, 'nodes: ', meas_dict['nodes'], 'edges: ', meas_dict['edges'], 'triangles: ',
              meas_dict['triangles'], 'diameter: ', meas_dict['diameter'], 'components: ', meas_dict['components'],
              'communities: ', meas_dict['communities'], 'coeff: ', meas_dict['coeff'], 'mean: ', m, 'std: ', std)

        plt.plot(pl)
        plt.title(f'r = {r} ,q = {q},k = {k}')
        plt.savefig(f'degree_distribution/ws_{r}_{q}_{k}.jpg')
        plt.clf()

    df = pd.DataFrame(temp_df)
    df.to_csv(f'ws_q_{q}.csv', index=False)
