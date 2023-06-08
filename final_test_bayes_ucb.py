import sys

import numpy as np
import json

sys.path.insert(0, 'task3/')
sys.path.insert(0, 'task2/')
from collections import Counter
import math
import time
import networkx as nx
import random
import pickle

import pandas as pd
from lesson5 import GenWS2DG, randomG
from SocNetMec import SocNetMec_Bayesian_UCB, SocNetMec_Bayesian_UCB_mudan, SocNetMec_UCB, SocNetMec_EPS, SocNetMec_TH
from SocNetMec_mixed import *
from task3.multi_diffusion_auction import *
from task3.vcg import auction
import matplotlib
import matplotlib.pyplot as plt
from utils import *
from task2.page_rank import pageRank
from task2.centrality import degree
from task2.voterank import voterank
from task2.shapley import shapley_closeness,shapley_degree,shapley_threshold, positive_decr_fun
from tqdm import tqdm

def input_data():
    n = 100
    r = 2.7
    kw = 1
    q = 4
    #G = GenWS2DG(n,r,kw,q) #This will be updated to the network model of net_x
    G = create_graph_from_txt('data/net_4.txt', sep=' ', directed=False)
    #G = randomG(n,p=0.3)
    n_nodes = G.number_of_nodes()
    k = math.floor(random.uniform(0.001*n_nodes, 0.05*n_nodes))
    k = 20
    T = 20000 # tra 20000 e 200000

    arms_set = [v for v in G.nodes()]
    """
    cen = pageRank(G)
    arms_set = list(top(G,cen,200).keys())
    """

    # Is a dictonary with whose keys are strings representing the different type of auctions and whose 
    # value is a list: truthful bidding, truthful reporting, function 
    auctions={} 
    auctions["MUDAN"] = [True, True, auction_mudan]
    auctions["MUDAR"] = [True, False, auction_mudar]
    auctions["VCG"] = [True, True, auction]

    #for the oracle val
    print("Val oracle: ")
    val = dict()
    for t in tqdm(range(T)):
        val[t] = dict()
        for u in G.nodes():
            val[t][u] = random.randint(1, 100)

    print("Val oracle done")
    
    #for the oracle prob
    print("Prob oracle: ")
    p = dict()
    for u in G.nodes():
        p[u] = dict()
    for u in tqdm(G.nodes()):
        for v in G[u]:
            if v not in p[u]:
                t=min(0.25, 2/max(G.degree(u), G.degree(v)))
                p[u][v] = p[v][u] = random.uniform(0, t)
                #p[u][v] = p[v][u] = 0.25
                G[u][v]['weight'] = round(p[u][v], 5)
    
    print("Prop oracle done")
            
    return G, k, T, val, p, arms_set, auctions

def prob(u, v):
    #print(v, " accepts the invitation from ",u," : ")
    r = random.random()
    #print("r: ", r)
    #print("p: ", p[u][v])
    if r <= p[u][v]:
        return True
    return False

def valf(t, u):
    return val[t][u]

def give_eps(t):
       if t == 0:
           return 1  #for the first step we cannot make exploitation, so eps_1 = 1
       return (len(arms_set)*math.log(t+1)/(t+1))**(1/3)

# Funzione per inizializzare le priori dei bracci utilizzando la centralità e la varianza scalata
def initialize_arms_prior(centrality, arms_set):
    max_centrality = max(np.array(list(centrality.values())))
    print(max_centrality)
    norm_centrality = normalize_centrality(centrality)
    arms_prior = {a:(norm_centrality[a],scale_variance(centrality[a], max_centrality)) for a in arms_set}
    return arms_prior

# Funzione per scalare il valore della varianza in base alla centralità
def scale_variance(val, max_val):
    max_variance = 2 
    scaled_variance = max_variance * (1 - val / max_val)
    return scaled_variance

def normalize_centrality(centrality):
    values = np.array(list(centrality.values()))
    mean = np.mean(values)
    std = np.std(values)
    print(mean,std)
    normalized = {a:(centrality[a] - mean) / std for a in centrality.keys()}
    return normalized

G, k, T, val, p, arms_set, auctions = input_data()

print("pagerank start")
"""cen = pageRank(G)
with open('pageRank.pickle', 'wb') as handle:
    pickle.dump(cen, handle, protocol=pickle.HIGHEST_PROTOCOL)"""

with open('pageRank.pickle', 'rb') as handle:
    cen = pickle.load(handle)
print("Pagerank done")
arms_set_page_rank = list(top(G,cen,200).keys())
gaussian_dist_pagerank = initialize_arms_prior(cen,arms_set_page_rank)

print("degree start")
"""cen = degree(G)
with open('degree.pickle', 'wb') as handle:
    pickle.dump(cen, handle, protocol=pickle.HIGHEST_PROTOCOL)"""

with open('degree.pickle', 'rb') as handle:
    cen = pickle.load(handle)

print("degree done")
arms_set_degree = list(top(G,cen,200).keys())
gaussian_dist_degree = initialize_arms_prior(cen,arms_set_degree)

"""
print("voterank start")
cen = voterank(G)
with open('voterank.pickle', 'wb') as handle:
    pickle.dump(cen, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('voterank.pickle', 'rb') as handle:
    cen = pickle.load(handle)

print("voterank done")
arms_set_voterank = list(top(G,cen,200).keys())
"""

print("shapley_degree start")
"""cen = shapley_degree(G)
with open('shapley_degree.pickle', 'wb') as handle:
    pickle.dump(cen, handle, protocol=pickle.HIGHEST_PROTOCOL)"""

with open('shapley_degree.pickle', 'rb') as handle:
    cen = pickle.load(handle)

print("shapley_degree done")
arms_set_shapley_degree = list(top(G,cen,200).keys())
gaussian_dist_shapley_degree = initialize_arms_prior(cen,arms_set_shapley_degree)



print("shapley_threshold start")
"""cen = shapley_threshold(G)
with open('shapley_threshold.pickle', 'wb') as handle:
    pickle.dump(cen, handle, protocol=pickle.HIGHEST_PROTOCOL)"""

with open('shapley_threshold.pickle', 'rb') as handle:
    cen = pickle.load(handle)
print("shapley_threshold done")
arms_set_shapley_threshold = list(top(G,cen,200).keys())
gaussian_dist_shapley_threshold = initialize_arms_prior(cen,arms_set_shapley_threshold)

print("shapley_closeness start")
"""cen = shapley_closeness(G,positive_decr_fun)
with open('shapley_closeness.pickle', 'wb') as handle:
    pickle.dump(cen, handle, protocol=pickle.HIGHEST_PROTOCOL)"""

with open('shapley_closeness.pickle', 'rb') as handle:
    cen = pickle.load(handle)
print("shapley_closeness done")
arms_set_shapley_closeness= list(top(G,cen,200).keys())
gaussian_dist_shapley_closeness = initialize_arms_prior(cen,arms_set_shapley_closeness)




"""fig = matplotlib.pyplot.figure()
edge_labels = nx.get_edge_attributes(G, "weight")
pos=nx.spring_layout(G, seed=7)
# nodes
nx.draw_networkx_nodes(G, pos, node_size=300)
nx.draw_networkx_edges(G, pos, width=1)
# node labels
nx.draw_networkx_labels(G, pos, font_size=6, font_family="sans-serif")
# edge weight labels
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()"""
listk = [1,4,20,50]
dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
for k in listk:
    snm_ucb_mudar=SocNetMec_UCB(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_page_rank)
    ucb_mudar_revenue = 0
    opt_ucb_mudar_reward = 0
    regrets_ucb_mudar = {}

    tic = time.time()
    for step in range(T):
        ucb_mudar_revenue += snm_ucb_mudar.run(step, prob, valf)
        opt_ucb_mudar_reward += snm_ucb_mudar.get_best_arm_approx()
        regrets_ucb_mudar[step] = opt_ucb_mudar_reward - ucb_mudar_revenue
    toc = time.time()
    exe_time_ucb_mudar = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("UCB_pr")
    dict_results["Auction"].append("MUDAR")
    dict_results["Time (s)"].append(exe_time_ucb_mudar)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudar_revenue)


    snm_ucb_mudar=SocNetMec_UCB(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_degree)
    ucb_mudar_revenue = 0
    opt_ucb_mudar_reward = 0
    regrets_ucb_mudar = {}

    tic = time.time()
    for step in range(T):
        ucb_mudar_revenue += snm_ucb_mudar.run(step, prob, valf)
        opt_ucb_mudar_reward += snm_ucb_mudar.get_best_arm_approx()
        regrets_ucb_mudar[step] = opt_ucb_mudar_reward - ucb_mudar_revenue
    toc = time.time()
    exe_time_ucb_mudar = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("UCB_deg")
    dict_results["Auction"].append("MUDAR")
    dict_results["Time (s)"].append(exe_time_ucb_mudar)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudar_revenue)

    snm_ucb_mudar=SocNetMec_UCB(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_shapley_degree)
    ucb_mudar_revenue = 0
    opt_ucb_mudar_reward = 0
    regrets_ucb_mudar = {}

    tic = time.time()
    for step in range(T):
        ucb_mudar_revenue += snm_ucb_mudar.run(step, prob, valf)
        opt_ucb_mudar_reward += snm_ucb_mudar.get_best_arm_approx()
        regrets_ucb_mudar[step] = opt_ucb_mudar_reward - ucb_mudar_revenue
    toc = time.time()
    exe_time_ucb_mudar = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("UCB_s_deg")
    dict_results["Auction"].append("MUDAR")
    dict_results["Time (s)"].append(exe_time_ucb_mudar)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudar_revenue)

    snm_ucb_mudar=SocNetMec_UCB(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_shapley_threshold)
    ucb_mudar_revenue = 0
    opt_ucb_mudar_reward = 0
    regrets_ucb_mudar = {}

    tic = time.time()
    for step in range(T):
        ucb_mudar_revenue += snm_ucb_mudar.run(step, prob, valf)
        opt_ucb_mudar_reward += snm_ucb_mudar.get_best_arm_approx()
        regrets_ucb_mudar[step] = opt_ucb_mudar_reward - ucb_mudar_revenue
    toc = time.time()
    exe_time_ucb_mudar = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("UCB_s_th")
    dict_results["Auction"].append("MUDAR")
    dict_results["Time (s)"].append(exe_time_ucb_mudar)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudar_revenue)


    snm_ucb_mudar=SocNetMec_UCB(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_shapley_closeness)
    ucb_mudar_revenue = 0
    opt_ucb_mudar_reward = 0
    regrets_ucb_mudar = {}

    tic = time.time()
    for step in range(T):
        ucb_mudar_revenue += snm_ucb_mudar.run(step, prob, valf)
        opt_ucb_mudar_reward += snm_ucb_mudar.get_best_arm_approx()
        regrets_ucb_mudar[step] = opt_ucb_mudar_reward - ucb_mudar_revenue
    toc = time.time()
    exe_time_ucb_mudar = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("UCB_s_cl")
    dict_results["Auction"].append("MUDAR")
    dict_results["Time (s)"].append(exe_time_ucb_mudar)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudar_revenue)

    df1 = pd.DataFrame(dict_results, columns=['Bandit','Auction', 'Time (s)', 'T', 'k', 'Revenue'])
    df1.to_csv("final_bayes_mudar_g.csv", index=False)


for k in listk:
    snm_ucb_mudar=SocNetMec_Bayesian_UCB(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_page_rank, gaussian_dist=gaussian_dist_pagerank)
    ucb_mudar_revenue = 0
    opt_ucb_mudar_reward = 0
    regrets_ucb_mudar = {}

    tic = time.time()
    for step in range(T):
        ucb_mudar_revenue += snm_ucb_mudar.run(step, prob, valf)
        opt_ucb_mudar_reward += snm_ucb_mudar.get_best_arm_approx()
        regrets_ucb_mudar[step] = opt_ucb_mudar_reward - ucb_mudar_revenue
    toc = time.time()
    exe_time_ucb_mudar = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("Bayesian_UCB_pr")
    dict_results["Auction"].append("MUDAR")
    dict_results["Time (s)"].append(exe_time_ucb_mudar)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudar_revenue)


    snm_ucb_mudar=SocNetMec_Bayesian_UCB(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_degree, gaussian_dist=gaussian_dist_degree)
    ucb_mudar_revenue = 0
    opt_ucb_mudar_reward = 0
    regrets_ucb_mudar = {}

    tic = time.time()
    for step in range(T):
        ucb_mudar_revenue += snm_ucb_mudar.run(step, prob, valf)
        opt_ucb_mudar_reward += snm_ucb_mudar.get_best_arm_approx()
        regrets_ucb_mudar[step] = opt_ucb_mudar_reward - ucb_mudar_revenue
    toc = time.time()
    exe_time_ucb_mudar = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("Bayesian_UCB_deg")
    dict_results["Auction"].append("MUDAR")
    dict_results["Time (s)"].append(exe_time_ucb_mudar)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudar_revenue)

    snm_ucb_mudar=SocNetMec_Bayesian_UCB(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_shapley_degree, gaussian_dist=gaussian_dist_shapley_degree)
    ucb_mudar_revenue = 0
    opt_ucb_mudar_reward = 0
    regrets_ucb_mudar = {}

    tic = time.time()
    for step in range(T):
        ucb_mudar_revenue += snm_ucb_mudar.run(step, prob, valf)
        opt_ucb_mudar_reward += snm_ucb_mudar.get_best_arm_approx()
        regrets_ucb_mudar[step] = opt_ucb_mudar_reward - ucb_mudar_revenue
    toc = time.time()
    exe_time_ucb_mudar = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("Bayesian_UCB_s_deg")
    dict_results["Auction"].append("MUDAR")
    dict_results["Time (s)"].append(exe_time_ucb_mudar)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudar_revenue)

    snm_ucb_mudar=SocNetMec_Bayesian_UCB(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_shapley_threshold, gaussian_dist=gaussian_dist_shapley_threshold)
    ucb_mudar_revenue = 0
    opt_ucb_mudar_reward = 0
    regrets_ucb_mudar = {}

    tic = time.time()
    for step in range(T):
        ucb_mudar_revenue += snm_ucb_mudar.run(step, prob, valf)
        opt_ucb_mudar_reward += snm_ucb_mudar.get_best_arm_approx()
        regrets_ucb_mudar[step] = opt_ucb_mudar_reward - ucb_mudar_revenue
    toc = time.time()
    exe_time_ucb_mudar = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("Bayesian_UCB_s_th")
    dict_results["Auction"].append("MUDAR")
    dict_results["Time (s)"].append(exe_time_ucb_mudar)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudar_revenue)


    snm_ucb_mudar=SocNetMec_Bayesian_UCB(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_shapley_closeness, gaussian_dist=gaussian_dist_shapley_closeness)
    ucb_mudar_revenue = 0
    opt_ucb_mudar_reward = 0
    regrets_ucb_mudar = {}

    tic = time.time()
    for step in range(T):
        ucb_mudar_revenue += snm_ucb_mudar.run(step, prob, valf)
        opt_ucb_mudar_reward += snm_ucb_mudar.get_best_arm_approx()
        regrets_ucb_mudar[step] = opt_ucb_mudar_reward - ucb_mudar_revenue
    toc = time.time()
    exe_time_ucb_mudar = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("Bayesian_UCB_s_cl")
    dict_results["Auction"].append("MUDAR")
    dict_results["Time (s)"].append(exe_time_ucb_mudar)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudar_revenue)

    df1 = pd.DataFrame(dict_results, columns=['Bandit','Auction', 'Time (s)', 'T', 'k', 'Revenue'])
    df1.to_csv("final_bayes_mudar_g.csv", index=False)







dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}


for k in listk:
    snm_ucb_mudan=SocNetMec_UCB_mudan(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_page_rank)
    ucb_mudan_revenue = 0
    opt_ucb_mudan_reward = 0
    regrets_ucb_mudan = {}

    tic = time.time()
    for step in range(T):
        ucb_mudan_revenue += snm_ucb_mudan.run(step, prob, valf)
        opt_ucb_mudan_reward += snm_ucb_mudan.get_best_arm_approx()
        regrets_ucb_mudan[step] = opt_ucb_mudan_reward - ucb_mudan_revenue
    toc = time.time()
    exe_time_ucb_mudan = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("UCB_pr")
    dict_results["Auction"].append("MUDAN")
    dict_results["Time (s)"].append(exe_time_ucb_mudan)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudan_revenue)


    snm_ucb_mudan=SocNetMec_UCB_mudan(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_degree)
    ucb_mudan_revenue = 0
    opt_ucb_mudan_reward = 0
    regrets_ucb_mudan = {}

    tic = time.time()
    for step in range(T):
        ucb_mudan_revenue += snm_ucb_mudan.run(step, prob, valf)
        opt_ucb_mudan_reward += snm_ucb_mudan.get_best_arm_approx()
        regrets_ucb_mudan[step] = opt_ucb_mudan_reward - ucb_mudan_revenue
    toc = time.time()
    exe_time_ucb_mudan = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("UCB_deg")
    dict_results["Auction"].append("MUDAN")
    dict_results["Time (s)"].append(exe_time_ucb_mudan)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudan_revenue)

    snm_ucb_mudan=SocNetMec_UCB_mudan(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_shapley_degree)
    ucb_mudan_revenue = 0
    opt_ucb_mudan_reward = 0
    regrets_ucb_mudan = {}

    tic = time.time()
    for step in range(T):
        ucb_mudan_revenue += snm_ucb_mudan.run(step, prob, valf)
        opt_ucb_mudan_reward += snm_ucb_mudan.get_best_arm_approx()
        regrets_ucb_mudan[step] = opt_ucb_mudan_reward - ucb_mudan_revenue
    toc = time.time()
    exe_time_ucb_mudan = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("UCB_s_deg")
    dict_results["Auction"].append("MUDAN")
    dict_results["Time (s)"].append(exe_time_ucb_mudan)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudan_revenue)

    snm_ucb_mudan=SocNetMec_UCB_mudan(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_shapley_threshold)
    ucb_mudan_revenue = 0
    opt_ucb_mudan_reward = 0
    regrets_ucb_mudan = {}

    tic = time.time()
    for step in range(T):
        ucb_mudan_revenue += snm_ucb_mudan.run(step, prob, valf)
        opt_ucb_mudan_reward += snm_ucb_mudan.get_best_arm_approx()
        regrets_ucb_mudan[step] = opt_ucb_mudan_reward - ucb_mudan_revenue
    toc = time.time()
    exe_time_ucb_mudan = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("UCB_s_th")
    dict_results["Auction"].append("MUDAN")
    dict_results["Time (s)"].append(exe_time_ucb_mudan)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudan_revenue)


    snm_ucb_mudan=SocNetMec_UCB_mudan(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_shapley_closeness)
    ucb_mudan_revenue = 0
    opt_ucb_mudan_reward = 0
    regrets_ucb_mudan = {}

    tic = time.time()
    for step in range(T):
        ucb_mudan_revenue += snm_ucb_mudan.run(step, prob, valf)
        opt_ucb_mudan_reward += snm_ucb_mudan.get_best_arm_approx()
        regrets_ucb_mudan[step] = opt_ucb_mudan_reward - ucb_mudan_revenue
    toc = time.time()
    exe_time_ucb_mudan = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("UCB_s_cl")
    dict_results["Auction"].append("MUDAN")
    dict_results["Time (s)"].append(exe_time_ucb_mudan)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudan_revenue)

    df1 = pd.DataFrame(dict_results, columns=['Bandit','Auction', 'Time (s)', 'T', 'k', 'Revenue'])
    df1.to_csv("final_bayes_mudan_g.csv", index=False)


for k in listk:
    snm_ucb_mudan=SocNetMec_Bayesian_UCB_mudan(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_page_rank, gaussian_dist=gaussian_dist_pagerank)
    ucb_mudan_revenue = 0
    opt_ucb_mudan_reward = 0
    regrets_ucb_mudan = {}

    tic = time.time()
    for step in range(T):
        ucb_mudan_revenue += snm_ucb_mudan.run(step, prob, valf)
        opt_ucb_mudan_reward += snm_ucb_mudan.get_best_arm_approx()
        regrets_ucb_mudan[step] = opt_ucb_mudan_reward - ucb_mudan_revenue
    toc = time.time()
    exe_time_ucb_mudan = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("Bayesian_UCB_pr")
    dict_results["Auction"].append("MUDAN")
    dict_results["Time (s)"].append(exe_time_ucb_mudan)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudan_revenue)


    snm_ucb_mudan=SocNetMec_Bayesian_UCB_mudan(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_degree, gaussian_dist=gaussian_dist_degree)
    ucb_mudan_revenue = 0
    opt_ucb_mudan_reward = 0
    regrets_ucb_mudan = {}

    tic = time.time()
    for step in range(T):
        ucb_mudan_revenue += snm_ucb_mudan.run(step, prob, valf)
        opt_ucb_mudan_reward += snm_ucb_mudan.get_best_arm_approx()
        regrets_ucb_mudan[step] = opt_ucb_mudan_reward - ucb_mudan_revenue
    toc = time.time()
    exe_time_ucb_mudan = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("Bayesian_UCB_deg")
    dict_results["Auction"].append("MUDAN")
    dict_results["Time (s)"].append(exe_time_ucb_mudan)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudan_revenue)

    snm_ucb_mudan=SocNetMec_Bayesian_UCB_mudan(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_shapley_degree, gaussian_dist=gaussian_dist_shapley_degree)
    ucb_mudan_revenue = 0
    opt_ucb_mudan_reward = 0
    regrets_ucb_mudan = {}

    tic = time.time()
    for step in range(T):
        ucb_mudan_revenue += snm_ucb_mudan.run(step, prob, valf)
        opt_ucb_mudan_reward += snm_ucb_mudan.get_best_arm_approx()
        regrets_ucb_mudan[step] = opt_ucb_mudan_reward - ucb_mudan_revenue
    toc = time.time()
    exe_time_ucb_mudan = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("Bayesian_UCB_s_deg")
    dict_results["Auction"].append("MUDAN")
    dict_results["Time (s)"].append(exe_time_ucb_mudan)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudan_revenue)

    snm_ucb_mudan=SocNetMec_Bayesian_UCB_mudan(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_shapley_threshold, gaussian_dist=gaussian_dist_shapley_threshold)
    ucb_mudan_revenue = 0
    opt_ucb_mudan_reward = 0
    regrets_ucb_mudan = {}

    tic = time.time()
    for step in range(T):
        ucb_mudan_revenue += snm_ucb_mudan.run(step, prob, valf)
        opt_ucb_mudan_reward += snm_ucb_mudan.get_best_arm_approx()
        regrets_ucb_mudan[step] = opt_ucb_mudan_reward - ucb_mudan_revenue
    toc = time.time()
    exe_time_ucb_mudan = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("Bayesian_UCB_s_th")
    dict_results["Auction"].append("MUDAN")
    dict_results["Time (s)"].append(exe_time_ucb_mudan)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudan_revenue)


    snm_ucb_mudan=SocNetMec_Bayesian_UCB_mudan(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set_shapley_closeness, gaussian_dist=gaussian_dist_shapley_closeness)
    ucb_mudan_revenue = 0
    opt_ucb_mudan_reward = 0
    regrets_ucb_mudan = {}

    tic = time.time()
    for step in range(T):
        ucb_mudan_revenue += snm_ucb_mudan.run(step, prob, valf)
        opt_ucb_mudan_reward += snm_ucb_mudan.get_best_arm_approx()
        regrets_ucb_mudan[step] = opt_ucb_mudan_reward - ucb_mudan_revenue
    toc = time.time()
    exe_time_ucb_mudan = round(toc - tic, 3)

    #dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}
    dict_results["Bandit"].append("Bayesian_UCB_s_cl")
    dict_results["Auction"].append("MUDAN")
    dict_results["Time (s)"].append(exe_time_ucb_mudan)
    dict_results["T"].append(T)
    dict_results["k"].append(k)
    dict_results["Revenue"].append(ucb_mudan_revenue)

    df1 = pd.DataFrame(dict_results, columns=['Bandit','Auction', 'Time (s)', 'T', 'k', 'Revenue'])
    df1.to_csv("final_bayes_mudan_g.csv", index=False)

print("END")