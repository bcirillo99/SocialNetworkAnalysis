import sys

import pandas as pd

from idm import GIDM
from multi_diffusion_auction import auction_mudan

sys.path.insert(0, '../')
from utils import create_auction,auction_results
import networkx as nx
import random
from collections import deque
from vcg import auction
import time
from model import GenWS2DG
from multi_diffusion_auction import auction_mudar

from argparse import ArgumentParser



if __name__ == '__main__':
    # Test e Analisi delle tempistiche

    parser = ArgumentParser()
    parser.add_argument('--n', help='numero di nodi', type=int, default=1000)
    parser.add_argument('--r', help='raggio per modello Watts-Strogatz', type=float, default=2.71)
    parser.add_argument('--kw', help='numero di weak ties per modello Watts-Strogatz', type=int, default=1)
    parser.add_argument('--q', help='parametro q per modello Watts-Strogatz', type=float, default=4.)
    parser.add_argument('--k', help='numero di elementi da vendere', type=int, default=10)
    parser.add_argument('--file_name', help='file dove salvare risultati', type=str, default="results_2.csv")
    
    args = parser.parse_args()

    n = args.n
    kw = args.kw
    r = args.r
    q = args.q
    k = args.k

    #df = pd.DataFrame()
    for k in [13,21,35]:
            
        dict_G1 = {'Auction':[], 'Time (s)':[], 'Sw':[], 'Rw': []}
        seller_net, reports, bids = create_auction(n,r,kw,q)
        
        
        ################################### MUDAR ###################################
        print("MUDAR")
        tic = time.time()
        allocations, payments = auction_mudar(k, seller_net, reports, bids)
        toc = time.time()
        exe_time = round(toc - tic, 3)

        sw,rw = auction_results(allocations,bids,payments)

        dict_G1["Auction"].append("MUDAR")
        dict_G1["Time (s)"].append(exe_time)
        dict_G1["Sw"].append(sw)
        dict_G1["Rw"].append(rw)

        ################################### MUDAN ###################################
        print("MUDAN")
        tic = time.time()
        allocations, payments = auction_mudan(k, seller_net, reports, bids)
        toc = time.time()
        exe_time = round(toc - tic, 3)

        sw,rw = auction_results(allocations,bids,payments)

        dict_G1["Auction"].append("MUDAN")
        dict_G1["Time (s)"].append(exe_time)
        dict_G1["Sw"].append(sw)
        dict_G1["Rw"].append(rw)

        ################################### VCG ###################################
        print("VCG")
        tic = time.time()
        allocations, payments = auction(k, seller_net, reports, bids)
        toc = time.time()
        exe_time = round(toc - tic, 3)

        sw,rw = auction_results(allocations,bids,payments)

        dict_G1["Auction"].append("VCG")
        dict_G1["Time (s)"].append(exe_time)
        dict_G1["Sw"].append(sw)
        dict_G1["Rw"].append(rw)

        ################################### GIDM ###################################
        print("GIDM")
        tic = time.time()
        allocations, payments = GIDM(k, seller_net, reports, bids)
        toc = time.time()
        exe_time = round(toc - tic, 3)

        sw,rw = auction_results(allocations,bids,payments)

        dict_G1["Auction"].append("GIDM")
        dict_G1["Time (s)"].append(exe_time)
        dict_G1["Sw"].append(sw)
        dict_G1["Rw"].append(rw)

        df1 = pd.DataFrame(dict_G1)
        print(df1)
        df1.to_csv("results_"+str(k)+".csv", index=False)
