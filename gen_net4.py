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
import matplotlib
import matplotlib.pyplot as plt
from utils import *
from task2.page_rank import pageRank
from task2.centrality import degree
from task2.voterank import voterank
from task2.shapley import shapley_closeness,shapley_degree,shapley_threshold, positive_decr_fun
from tqdm import tqdm

n = 20000
r = 2.71
kw = 1
q = 4


G = GenWS2DG(n,r,kw,q) #This will be updated to the network model of net_x
#G = create_graph_from_txt('data/tr.txt', sep=' ', directed=False)

nx.write_edgelist(G, "data/GenWS2DG_1.txt",data=False)

G = GenWS2DG(n,r,kw,q) #This will be updated to the network model of net_x
#G = create_graph_from_txt('data/tr.txt', sep=' ', directed=False)

nx.write_edgelist(G, "data/GenWS2DG_2.txt",data=False)