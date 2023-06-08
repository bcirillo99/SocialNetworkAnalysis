import ast
import copy

import networkx as nx
from collections import deque
import random
from tqdm import tqdm
from multi_diffusion_auction import Bidder


def auction(k: int, seller_net: set, reports: dict, bids: dict):
    """
    :param k: intero che rappresenta il numero di elementi omogenei da vendere
    :param seller_net: bidder collegati direttamente al seller
    :param reports: dizionario che ha come chiavi i
    bidder e come valori una lista di bidder a cui riportano le informazioni dell'asta
    :param bids: dizionario che ha  come chiavi differenti bidder e come valori le singole offerte
    :return: due dizionari allocation e payment. Allocation ha come chiavi i bidder e come valore un booleano, che assume
    valore True nel caso in cui il bidder ha ottenuto l'elemento altrimenti è False. Payment ha come chiavi i bidder e come
    valori il prezzo pagato dal bidder. Nel caso in cui il pagamento è negativo rappresenta la ricompensa da parte del seller.
    """

    payments = {}
    G = nx.DiGraph()

    # costruzione del grafo
    for bidder in seller_net:
        G.add_edge('seller', bidder)

    for key in reports.keys():
        for value in reports[key]:
            G.add_edge(key, value)

    bidder_list = [Bidder(bidder, bids[bidder]) for bidder in list(bids.keys())]
    bidder_list.sort(reverse=True)
    # social welfare dell'allocazione efficiente
    sw, allocation = simulate_auction2(k, list(bids.keys()), bids)

    # calcolo dei prezzi per ogni bidder
    for bidder, bid in bids.items():
        # ricerca degli offerenti che ricevono le informazioni sull'asta in assenza del bidder
        reachable = information_diffusion(G, bidder)
        # calcolo del social welfare escludendo il bidder vincente
        sw1, _ = simulate_auction2(k, reachable, bids)

        if not allocation[bidder]:
            bid = 0
        # calcolo del social welfare escludendo il bidder e l'item
        sw2 = sw - bid
        # regola di pagamento
        payments[bidder] = sw1 - sw2
    return allocation, payments


def simulate_auction(k, bidders: list, bids: dict):
    allocation = {bidder: False for bidder in bidders}
    # social welfare
    sw = 0.
    # nel caso in cui ci sono più bidder che offrono l'offerta più alta l'algoritmo
    # sceglie casualmente il bidder vincente
    random.shuffle(bidders)
    for item in range(k):

        winning_bidder = None
        winning_bid = 0.

        for bidder in bidders:
            if not allocation[bidder]:

                bid = bids[bidder]
                if winning_bid < bid:
                    winning_bid = bid
                    winning_bidder = bidder

        sw += winning_bid
        if winning_bidder is not None:
            allocation[winning_bidder] = True

    return sw, allocation

def critical_sequence(G: nx.DiGraph,node,root):
    cs = [node]
    if node!=root:
        while len(list(G.predecessors(node))) == 1 and list(G.predecessors(node))[0]!=root:
            node = list(G.predecessors(node))[0]
            cs.append(node)
    return cs

def simulate_auction2(k, bidders: list, bids: dict):
    bidder_list = [Bidder(bidder, bids[bidder]) for bidder in bidders]
    bidder_list.sort(reverse=True)
    winners = bidder_list[:k]
    
    sw = 0
    allocations = {bidder.name: False for bidder in bidder_list}
    for winning_bidder in winners:
        sw += winning_bidder.bid
        allocations[winning_bidder.name] = True
    return sw, allocations


def information_diffusion(G: nx.DiGraph, bidder, seller='seller'):
    """
    Restituisce la lista di bidder raggiungibili a partire dal nodo 'seller' escludendo il nodo 'bidder'
    :param G: Grafo orientato della libreria networkx
    :param bidder: nodo bidder
    :param seller: nodo seller
    :return: tutti gli offerenti raggiungibili dal seller in assenza della diffusione dell' informazione del nodo bidder
    """
    level = deque([seller])
    visited = [seller]

    while len(level) > 0:

        n = level.popleft()
        for c in G[n]:
            if (c not in visited) and (c != bidder):
                level.append(c)
                visited.append(c)
    # escludo il nodo seller
    return visited[1:]

def auction_results(allocations: dict, bids: dict, payments: dict):
    rw = sum(payments.values())
    sw = 0.
    for bidder, alloc in allocations.items():
        if alloc:
            sw += bids[bidder]
    print("sw: ",sw)
    print("rw: ",rw)
    return sw, rw

if __name__ == '__main__':

    
    seller_net = {'1'}
    reports ={'1': ['2', '3', '4', '5', '6'], '2': ['7'], '3': ['8'], '4': ['9'], '5': ['10', '11'], '6': ['12'], '10': ['13', '14', '15'], '12': ['16'], '14': ['17'], '17': ['18'], '18': ['19'], '19': ['20']}
    bids = {'1': 21, '2': 71, '3': 10, '4': 34, '5': 62, '6': 7, '7': 32, '8': 12, '9': 52, '10': 6, '11': 96, '12': 91, '13': 86, '14': 74, '15': 81, '16': 43, '17': 82, '18': 69, '19': 18, '20': 56}
    allocations, payments = auction(555, seller_net, reports, bids)
    print("K: ",555)
    print("\npayments:")
    print(payments)
    print("\nallocation:")
    print(allocations)
    auction_results(allocations,bids, payments)

    seller_net = {'1'}
    reports ={'1': ['2', '3', '4', '5', '6'], '2': ['7'], '3': ['8'], '4': ['9'], '5': ['10', '11'], '6': ['12'], '10': ['13', '14', '15'], '12': ['16'], '14': ['17'], '17': ['18'], '18': ['19'], '19': ['20']}
    bids = {'1': 21, '2': 71, '3': 10, '4': 34, '5': 62, '6': 7, '7': 32, '8': 12, '9': 52, '10': 6, '11': 96, '12': 91, '13': 86, '14': 74, '15': 81, '16': 43, '17': 82, '18': 69, '19': 18, '20': 56}
    allocations, payments = auction(18, seller_net, reports, bids)
    print("K: ",18)
    print("\npayments:")
    print(payments)
    print("\nallocation:")
    print(allocations)
    print(sum(payments.values()))
    auction_results(allocations,bids, payments)

    print("a" not in "a")

"""    with open("data.txt", "r") as f:
        seller_net = ast.literal_eval(f.readline())
        reports = ast.literal_eval(f.readline())
        bids = ast.literal_eval(f.readline())
        k = int(f.readline())



    allocations, payments = auction(k, seller_net, reports, bids)

    print(sum(payments.values()))
    auction_results(allocations,bids, payments)"""


