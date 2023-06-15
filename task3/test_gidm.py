import sys

from idm import GIDM
from utils import create_auction
sys.path.insert(0, '../')
import networkx as nx
import random
from collections import deque
from vcg import *
import time
from model import GenWS2DG
from multi_diffusion_auction import auction_mudar



if __name__ == '__main__':
    seller_net, reports, bids = create_auction(30)


    
    # MUDAR 
    print("GIDM")

    bad_case = 0
    good_case = 0
    maxbid = max(list(bids.values()))
    print("\nCase v<v\'\n:")
    for i in tqdm(range(100)):
        for node in bids.keys():
            realbid = bids[node]
            # Caso Base
            allocations, payments = GIDM(5, seller_net, reports, bids)
            

            true_alloc = allocations[node]
            if true_alloc:
                true_utility = realbid-payments[node]
            else:
                if payments[node] == 0:
                    true_utility = 0
                else:
                    true_utility = -payments[node]
            

            # Caso v'< real v
            new_bids = copy.deepcopy(bids)
            new_bids[node] = random.randint(1,realbid)
            allocations, payments = GIDM(5, seller_net, reports, new_bids)
            
            new_alloc = allocations[node]
            if new_alloc:
                new_utility = realbid-payments[node]
            else:
                if payments[node] == 0:
                    new_utility = 0
                else:
                    new_utility = -payments[node]
            
            if new_utility < true_utility:
                bad_case+=1
            elif new_utility == true_utility:
                if new_alloc and not true_alloc:
                    good_case+=1
                else: 
                    bad_case+=1
            else:
                good_case+=1
    
    if good_case < 1:
        print("truthfull bidding?: ", good_case>bad_case)
        print("good case: ",good_case)
        print("bad case: ",bad_case)


    bad_case = 0
    good_case = 0

    print("\nCase v>v\'\n:")
    for i in tqdm(range(100)):
        for node in bids.keys():

            realbid = bids[node]
            # Caso Base
            allocations, payments = GIDM(5, seller_net, reports, bids)
            

            true_alloc = allocations[node]
            if true_alloc:
                true_utility = realbid-payments[node]
            else:
                if payments[node] == 0:
                    true_utility = 0
                else:
                    true_utility = -payments[node]
            

            # Caso v'< real v
            new_bids = copy.deepcopy(bids)
            new_bids[node] = random.randint(realbid+1,realbid*2)
            allocations, payments = GIDM(5, seller_net, reports, new_bids)
            
            new_alloc = allocations[node]
            if new_alloc:
                new_utility = realbid-payments[node]
            else:
                if payments[node] == 0:
                    new_utility = 0
                else:
                    new_utility = -payments[node]
            
            if new_utility < true_utility:
                bad_case+=1
            elif new_utility == true_utility:
                if new_alloc and not true_alloc:
                    good_case+=1
                else: 
                    bad_case+=1
            else:
                good_case+=1
    if good_case < 1:
        print("truthfull bidding?: ", good_case>bad_case)
        print("good case: ",good_case)
        print("bad case: ",bad_case)