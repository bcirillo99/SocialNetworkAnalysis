import os
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
from SocNetMec import *
#from SocNetMec_mixed import *
from task3.multi_diffusion_auction import *
from task3.vcg import auction
from task3.idm import GIDM, generalize_diffusion_mechanism
import matplotlib
import matplotlib.pyplot as plt
from utils import *
from task2.page_rank import pageRank
from task2.centrality import degree
from task2.voterank import voterank
from task2.shapley import shapley_closeness,shapley_degree,shapley_threshold, positive_decr_fun
from tqdm import tqdm

def input_data(path_net=None):
    n = 100
    r = 2.7
    kw = 1
    q = 4
    #G = GenWS2DG(n,r,kw,q) #This will be updated to the network model of net_x
    if path_net is not None:
        G = create_graph_from_txt(path_net, sep=' ', directed=False)
        print("Rete generata con watts-strogatz")
    else:
        print("Net 4")
        G = create_graph_from_txt('data/net_4.txt', sep=' ', directed=False)
    #G = randomG(n,p=0.3)
    n_nodes = G.number_of_nodes()
    k = math.floor(random.uniform(0.001*n_nodes, 0.05*n_nodes))
    k = 20
    T = 80000 # tra 20000 e 200000

    arms_set = [v for v in G.nodes()]
    """
    cen = pageRank(G)
    arms_set = list(top(G,cen,200).keys())
    """

    # Is a dictonary with whose keys are strings representing the different type of auctions and whose 
    # value is a list: truthful bidding, truthful reporting, function 
    auctions={} 
    auctions["MUDAN"] = [True, True, auction_mudan]
    auctions["MUDAR"] = [False, True, auction_mudar]
    auctions["VCG"] = [True, True, auction]
    auctions["GIDM"] = [True, True, GIDM]

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
    normalized = {a:(centrality[a] - mean) / std for a in centrality.keys()}
    return normalized

G, k, T, val, p, arms_set, auctions = input_data()

####################################################################

################## CENTRALITY MEASURE COMPUTATION ##################

####################################################################

gaussian_dist_normal = {a:(0,1) for a in arms_set}



# Page Rank 

"""
cen = pageRank(G)
with open('pageRank.pickle', 'wb') as handle:
    pickle.dump(cen, handle, protocol=pickle.HIGHEST_PROTOCOL)"""

with open('pageRank.pickle', 'rb') as handle:
    cen = pickle.load(handle)

arms_set_page_rank = list(top(G,cen,200).keys()) # The 200 nodes with the highest centrality measure
gaussian_dist_pagerank = initialize_arms_prior(cen,arms_set_page_rank) # Mean and variance based on the centrality measure


# Vote Rank

"""
cen = voterank(G)
with open('voterank.pickle', 'wb') as handle:
    pickle.dump(cen, handle, protocol=pickle.HIGHEST_PROTOCOL)"""

with open('voterank.pickle', 'rb') as handle:
    cen = pickle.load(handle)

arms_set_vote_rank = list(top(G,cen,200).keys()) # The 200 nodes with the highest centrality measure
gaussian_dist_vote_rank = initialize_arms_prior(cen,arms_set_vote_rank) # Mean and variance based on the centrality measure

# Degree
"""
cen = degree(G)
with open('degree.pickle', 'wb') as handle:
    pickle.dump(cen, handle, protocol=pickle.HIGHEST_PROTOCOL)"""

with open('degree.pickle', 'rb') as handle:
    cen = pickle.load(handle)

arms_set_degree = list(top(G,cen,200).keys()) # The 200 nodes with the highest centrality measure
gaussian_dist_degree = initialize_arms_prior(cen,arms_set_degree) # Mean and variance based on the centrality measure


# shapley_degree
"""
cen = shapley_degree(G)
with open('shapley_degree.pickle', 'wb') as handle:
    pickle.dump(cen, handle, protocol=pickle.HIGHEST_PROTOCOL)"""

with open('shapley_degree.pickle', 'rb') as handle:
    cen = pickle.load(handle)

arms_set_shapley_degree = list(top(G,cen,200).keys()) # The 200 nodes with the highest centrality measure
gaussian_dist_shapley_degree = initialize_arms_prior(cen,arms_set_shapley_degree) # Mean and variance based on the centrality measure


# shapley_threshold
"""
cen = shapley_threshold(G)
with open('shapley_threshold.pickle', 'wb') as handle:
    pickle.dump(cen, handle, protocol=pickle.HIGHEST_PROTOCOL)"""

with open('shapley_threshold.pickle', 'rb') as handle:
    cen = pickle.load(handle)

arms_set_shapley_threshold = list(top(G,cen,200).keys()) # The 200 nodes with the highest centrality measure
gaussian_dist_shapley_threshold = initialize_arms_prior(cen,arms_set_shapley_threshold) # Mean and variance based on the centrality measure


# shapley_closeness

"""
cen = shapley_closeness(G,positive_decr_fun)
with open('shapley_closeness.pickle', 'wb') as handle:
    pickle.dump(cen, handle, protocol=pickle.HIGHEST_PROTOCOL)"""

with open('shapley_closeness.pickle', 'rb') as handle:
    cen = pickle.load(handle)

arms_set_shapley_closeness = list(top(G,cen,200).keys()) # The 200 nodes with the highest centrality measure
gaussian_dist_shapley_closeness = initialize_arms_prior(cen,arms_set_shapley_closeness) # Mean and variance based on the centrality measure




####################################################################

####################### Results Computation ########################

####################################################################


listk = [1,2,3,4,5] # Range of k values
list_auction = ["MUDAR"] # Different Auctions
dict_armset = {"pagerank": arms_set_page_rank, "degree": arms_set_degree, "voterank": arms_set_vote_rank}
dict_distributions = {"pagerank": gaussian_dist_pagerank, "degree": gaussian_dist_degree, "voterank": gaussian_dist_vote_rank}
dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[]}

########################## Normal armset ###########################

for auction in list_auction:
    dict_results =  {'Bandit':[],'Auction':[], 'Time (s)':[], 'T':[], 'k': [], "Revenue":[], "armset": []}
    for key in dict_armset.keys():
        arms_set = dict_armset[key]
        prior = dict_distributions[key]
        for k in listk:
            
            print("Auction: ",auction)
            print("k: ",k)
            print("armset: ", key)
            ##################         UCB           ##################

            snm_ucb=SocNetMec_UCB(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set, auction=auction)
            ucb_revenue = 0
            opt_ucb_reward = 0
            regrets_ucb = {}

            print("UCB: \n")
            tic = time.time()
            for step in tqdm(range(T)):
                ucb_revenue += snm_ucb.run(step, prob, valf)
                opt_ucb_reward += snm_ucb.get_best_arm_approx()
                regrets_ucb[step] = opt_ucb_reward - ucb_revenue
            toc = time.time()
            exe_time_ucb = round(toc - tic, 3)

            dict_results["Bandit"].append("UCB")
            dict_results["Auction"].append(auction)
            dict_results["Time (s)"].append(exe_time_ucb)
            dict_results["T"].append(T)
            dict_results["k"].append(k)
            dict_results["Revenue"].append(ucb_revenue)
            dict_results["armset"].append(key)

            print("total revenue: ", ucb_revenue)
            print()


            ##################       Bayesian UCB           ##################

            snm_ucb=SocNetMec_Bayesian_UCB(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set, auction=auction)
            ucb_revenue = 0
            opt_ucb_reward = 0
            regrets_ucb = {}

            print("Bayesian UCB: \n")
            tic = time.time()
            for step in tqdm(range(T)):
                ucb_revenue += snm_ucb.run(step, prob, valf)
                opt_ucb_reward += snm_ucb.get_best_arm_approx()
                regrets_ucb[step] = opt_ucb_reward - ucb_revenue
            toc = time.time()
            exe_time_ucb = round(toc - tic, 3)

            dict_results["Bandit"].append("Bayesian_UCB")
            dict_results["Auction"].append(auction)
            dict_results["Time (s)"].append(exe_time_ucb)
            dict_results["T"].append(T)
            dict_results["k"].append(k)
            dict_results["Revenue"].append(ucb_revenue)
            dict_results["armset"].append(key)

            print("total revenue: ", ucb_revenue)
            print()

            if key != "normal":
            ##################       Bayesian UCB prior           ##################

                snm_ucb=SocNetMec_Bayesian_UCB(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set, auction=auction, gaussian_dist=prior)
                ucb_revenue = 0
                opt_ucb_reward = 0
                regrets_ucb = {}

                print("Bayesian UCB prior: \n")
                tic = time.time()
                for step in tqdm(range(T)):
                    ucb_revenue += snm_ucb.run(step, prob, valf)
                    opt_ucb_reward += snm_ucb.get_best_arm_approx()
                    regrets_ucb[step] = opt_ucb_reward - ucb_revenue
                toc = time.time()
                exe_time_ucb = round(toc - tic, 3)

                dict_results["Bandit"].append("Bayesian_UCB_prior")
                dict_results["Auction"].append(auction)
                dict_results["Time (s)"].append(exe_time_ucb)
                dict_results["T"].append(T)
                dict_results["k"].append(k)
                dict_results["Revenue"].append(ucb_revenue)
                dict_results["armset"].append(key)

                print("total revenue: ", ucb_revenue)
                print()

            df1 = pd.DataFrame(dict_results, columns=['Bandit','Auction', 'Time (s)', 'T', 'k', 'Revenue','armset'])
            path = "final_results_prior2/"+str(T)
        
            if not os.path.exists(path):
                os.makedirs(path)
            df1.to_csv(path+"/"+auction+".csv", index=False)


print("\n\n\n\n\n\n\n\n\nEND\n\n\n\n\n\n\n\n\n")