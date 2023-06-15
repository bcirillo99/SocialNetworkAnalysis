import sys
sys.path.insert(0, '../')
import networkx as nx
import random
from collections import deque
from vcg import *
import time
from model import GenWS2DG
from task3.multi_diffusion_auction import auction_mudar
from utils import create_auction


if __name__ == '__main__':
    seller_net, reports, bids = create_auction(30)


    
    # MUDAR 
    print("MUDAR")

    bad_case = 0
    good_case = 0
    maxbid = max(list(bids.values()))
    print("\nCase v<v\'\n:")
    for i in tqdm(range(100)):
        for node in bids.keys():
            realbid = bids[node]
            # Caso Base
            allocations, payments = auction_mudar(5, seller_net, reports, bids)
            

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
            allocations, payments = auction_mudar(5, seller_net, reports, new_bids)
            
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
    print(bids)
    for i in tqdm(range(100)):
        for node in bids.keys():

            realbid = bids[node]
            # Caso Base
            allocations, payments = auction_mudar(5, seller_net, reports, bids)
            

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
            allocations, payments = auction_mudar(5, seller_net, reports, new_bids)
            
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
    else:
        print("truthfull bidding?: ", good_case>bad_case)
        print("good case: ",good_case)
        print("bad case: ",bad_case)





    """bad_case = 0
    good_case = 0
    print("\nCaso v'> v_max\n")
    for i in range(100):
        node = random.choice(list(bids.keys()))
        realbid = bids[node]
        # Caso Base
        allocations, payments = auction_mudar(5, seller_net, reports, bids)
        

        true_alloc = allocations[node]
        if true_alloc:
            true_utility = realbid-payments[node]
        else:
            if payments[node] == 0:
                true_utility = 0
            else:
                true_utility = -payments[node]
        

        # Caso v'> v_max
        new_bids = copy.deepcopy(bids)
        new_bids[node] = random.randint(maxbid+1,2*maxbid)
        allocations, payments = auction_mudar(5, seller_net, reports, new_bids)
        
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
    
    print("truthfull bidding?: ", good_case>bad_case)
    print("good case: ",good_case)
    print("bad case: ",bad_case)

    bad_case = 0
    good_case = 0
    print("\nCaso real v < v'< v_max\n")
    for i in range(100):
        node = random.choice(list(bids.keys()))
        realbid = bids[node]
        # Caso Base
        allocations, payments = auction_mudar(5, seller_net, reports, bids)
        true_alloc = allocations[node]
        if true_alloc:
            true_utility = realbid-payments[node]
        else:
            if payments[node] == 0:
                true_utility = 0
            else:
                true_utility = -payments[node]
        

        # Caso v'> v_max
        new_bids = copy.deepcopy(bids)
        new_bids[node] = random.randint(realbid,maxbid)
        allocations, payments = auction_mudar(5, seller_net, reports, new_bids)
        
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
    
    print("truthfull bidding?: ", good_case>bad_case)
    print("good case: ",good_case)
    print("bad case: ",bad_case)"""





"""


    # Caso v'> v_max
    print("\n\nCaso v'> v_max:\n\n")
    bids[node] = random.randint(maxbid+1,2*maxbid)
    print(bids)
    allocations, payments = auction_mudar(5, seller_net, reports, bids)
    auction_results(allocations,bids, payments)
    print("\npayments:")
    print(payments)
    print("\nallocation:")
    print(allocations)
    print("Utility per nodo selezionato: " + node)
    print("Allocation: ",allocations[node])
    if allocations[node]:
        print("Utility: ",realbid-payments[node])
    else:
        print("Utility: ", -payments[node])

    # Caso real v < v'< v_max
    print("\n\nCaso real v < v'< v_max:\n\n")
    bids[node] = random.randint(realbid,maxbid)
    print(bids)
    allocations, payments = auction_mudar(5, seller_net, reports, bids)
    auction_results(allocations,bids, payments)
    print("\npayments:")
    print(payments)
    print("\nallocation:")
    print(allocations)
    print("Utility per nodo selezionato: " + node)
    print("Allocation: ",allocations[node])
    if allocations[node]:
        print("Utility: ",realbid-payments[node])
    else:
        print("Utility: ", -payments[node])
    """

