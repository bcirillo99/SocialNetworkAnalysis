from argparse import ArgumentParser
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

def input_data(net4=True):
    n = 100
    r = 2.71
    kw = 1
    q = 4
    #G = GenWS2DG(n,r,kw,q) #This will be updated to the network model of net_x
    if net4:
        G = create_graph_from_txt('data/net_4.txt', sep=' ', directed=False)
        print("Net 4")
    else:
        G = GenWS2DG(n,r,kw,q) #This will be updated to the network model of net_x
        print("Rete generata con watts-strogatz")
    
    n_nodes = G.number_of_nodes()
    k = random.choice([2, 3, 4, 5])
    k = math.floor(random.uniform(0.001*n_nodes, 0.05*n_nodes))
    k = 20
    T = random.randint(20000, 200000)

    
    cen = pageRank(G)
    arms_set = list(top(G,cen,200).keys())
    

    # Is a dictonary with whose keys are strings representing the different type of auctions and whose 
    # value is a list: truthful bidding, truthful reporting, function 
    auctions={} 
    auctions["MUDAN"] = [True, True, auction_mudan]
    auctions["MUDAR"] = [False, True, auction_mudar]
    auctions["VCG"] = [True, True, auction]
    auctions["GIDM"] = [True, True, GIDM]

    auction="MUDAR"

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
            
    return G, k, T, val, p, arms_set, auctions, auction
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
           return 1  #for the first step we cannot make exploitation, so eps = 1
       return (len(arms_set)*math.log(t+1)/(t+1))**(1/3)


parser = ArgumentParser()
parser.add_argument('--T', help='numero di processi', type=int, default=20000)
args = parser.parse_args()



G, k, T, val, p, arms_set, auctions, auction = input_data()


####################################################################

####################### Results Computation ########################

####################################################################



##################         UCB           ##################

snm_ucb=SocNetMec_UCB(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set, auction=auction)
ucb_revenue = 0
opt_ucb_reward = 0
regrets_ucb = {}

print("UCB: \n")
for step in tqdm(range(T)):
    ucb_revenue += snm_ucb.run(step, prob, valf)
    opt_ucb_reward += snm_ucb.get_best_arm_approx()
    regrets_ucb[step] = opt_ucb_reward - ucb_revenue


print("total revenue: ", ucb_revenue)



print("\n\n\n\n\n\n\n\n\nEND\n\n\n\n\n\n\n\n\n")