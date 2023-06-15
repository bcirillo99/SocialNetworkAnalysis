from argparse import ArgumentParser
import os
import sys

import numpy as np
import json

sys.path.insert(0, 'task3/')
sys.path.insert(0, 'task2/')
from collections import Counter
import networkx as nx
import random

import pandas as pd
from model import GenWS2DG
from SocNetMec import *
from task3.multi_diffusion_auction import *
from task3.vcg import auction as auction_vcg
from task3.idm import GIDM, generalize_diffusion_mechanism
import matplotlib.pyplot as plt
from utils import *
from task2.page_rank import *
from task2.centrality import degree
from task2.voterank import *
from task2.shapley import *
from tqdm import tqdm

def input_data(n = 20000, r = 2.71, kw = 1, q = 4, rand = True):
    
    n = n #number of nodes
    r = r # the radius of each node (a node u is connected with each other node at distance at most r) - strong ties
    kw = kw # the number of random edges for each node u - weak ties
    q = q # is a term that evaluate how much the distance matters.
    
    if rand:
      G = GenWS2DG(n,r,kw,q) #This will be updated to the network model of net_x "Rete generata con watts-strogatz"
    else:
      G = create_graph_from_txt('data/net_4.txt', sep=' ', directed=False) #net4
    
    #n_nodes = G.number_of_nodes()
    #k = math.floor(random.uniform(0.001*n_nodes, 0.05*n_nodes))
    k = random.randint(2,5) # number of items to sell
    T = random.randint(20000, 200000) # horizon

    cen = pageRank(G)
    arms_set = list(top(G,cen,200).keys()) # as armset we choose to use the top 200 nodes with the highest centrality measure
    

    # Is a dictonary with whose keys are strings representing the different type of auctions and whose 
    # value is a list: truthful bidding, truthful reporting, function 
    auctions={} 
    auctions["MUDAN"] = [True, True, auction_mudan]
    auctions["MUDAR"] = [False, True, auction_mudar]
    auctions["VCG"] = [True, True, auction_vcg]
    auctions["GIDM"] = [True, True, GIDM]

    auction="GIDM"

    #for the oracle val
    print("Val oracle: ")
    val = dict()
    for t in tqdm(range(T)):
        val[t] = dict()
        for u in G.nodes():
            val[t][u] = random.randint(1, 100)
    
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

parser = ArgumentParser()
parser.add_argument('--n', help='number of nodes', type=int, default=20000)
parser.add_argument('--r', help='the radius of each node (a node u is connected with each other node at distance at most r) - strong ties', type=float, default=2.71)
parser.add_argument('--k', help='the number of random edges for each node u - weak ties', type=int, default=1)
parser.add_argument('--q', help='is a term that evaluate how much the distance matters', type=float, default=4.)
parser.add_argument('--rand', help='Parameter for deciding whether to use a watts-strogatz model or net4', action="store_false")
args = parser.parse_args()

n = args.n
r = args.r
k = args.k
q = args.q
rand = args.rand

print(rand)

G, k, T, val, p, arms_set, auctions, auction = input_data(n=n, kw=k, r=r, q=q, rand=rand)


####################################################################

####################### Results Computation ########################

####################################################################

snm_ucb=SocNetMec(G=G, T=T, k=k, auctions=auctions, arms_set=arms_set, auction=auction)
ucb_revenue = 0

for step in tqdm(range(T)):
    ucb_revenue += snm_ucb.run(step, prob, valf)

print("total revenue: ", ucb_revenue)