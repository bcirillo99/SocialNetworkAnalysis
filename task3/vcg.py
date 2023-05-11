import copy

import networkx as nx
from collections import deque
import random
from tqdm import tqdm

__all__ = ['auction']


def auction(k: int, seller_net: set, reports: dict, bids: dict):
    """
    Parameters
    ----------

    k: int
        is the number of item to sell;
    seller_net: set of strings
        is a set of strings each identifying a different bidder;
    reports: dict
        is a dictionary whose keys are strings each identifying a different bidder and whose 
        values are sets of strings representing the set of bidders to which the bidder identified 
        by the key reports the information about the auction;
    bids: dict
        is a dictionary whose keys are strings each identifying a different bidder and whose values
        are numbers defining the bid of the bidder identified by that key

    Returns
    -------
    allocation: dict
        that is a dictionary that has as keys the strings identifying each of the bidders
        that submitted a bid, and as value a boolean True if this bidder is allocated one of the items,
        and False otherwise.
    payments: dict
        that is a dictionary that has as keys the strings identifying each of the bidders that
        submitted a bid, and as value the price that she pays. Here, a positive price means that the
        bidder is paying to the seller, while a negative price means that the seller is paying to the bidder.
    """

    payments = {}
    G = nx.DiGraph()

    # costruzione del grafo
    for bidder in seller_net:
        G.add_edge('seller', bidder)

    for key in reports.keys():
        for value in reports[key]:
            G.add_edge(key, value)

    # social welfare dell'allocazione efficiente
    sw, allocation = simulate_auction(k, list(bids.keys()), bids)
    # calcolo dei prezzi per ogni bidder

    for bidder, bid in tqdm(bids.items()):
        # ricerca degli offerenti che ricevono le informazioni sull'asta in assenza di "bidder"
        reachable = information_diffusion(G, bidder)
        # calcolo del social welfare escludendo il bidder vincente
        sw1, _ = simulate_auction(k, reachable, bids)
        # calcolo del social welfare escludendo il bidder e l'item
        if not allocation[bidder]:
            bid = 0
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
            if (c not in visited) and (c is not bidder):
                level.append(c)
                visited.append(c)
    # escludo il nodo seller
    return visited[1:]


# SECOND PRICE AUCTION E VCG
bids = {'A': 2, 'B': 30, 'C': 27, 'D': 25, 'E': 12, 'F': 5, 'G': 2, 'S': 100, 'T': 10, 'H': 20}
seller_net = {'A', 'B', 'C', 'D', 'E', 'F', 'S', 'G', 'T', 'H'}
reports = {}
# Notiamo che il bidder S è il più alto offerente e paga quanto la seconda offerta più alta
print(auction(1, seller_net, reports, bids))

# EFFETTO DELLA DIFFUSIONE
# bids = {'A': 2, 'B': 30, 'C': 27, 'D': 25, 'E': 12, 'F': 5, 'G': 2, 'S': 100, 'T': 10, 'H': 20}
# seller_net = {'A', 'B', 'C', 'D', 'E', 'F', 'G','T','H'}
# reports = {'A':['S']}
# Notiamo che A non ottiene l'elemento ma riportando l'informazione a S (bidder vincente),
# ottiene una ricompensa dal seller
# print(auction(3, seller_net, reports, bids))

# PROPOSITION 1.
# VCG con diffusione dell'informazione non permette di ottenere un'alta ricompensa per il seller, in questo
# caso il seller è in perdita
# seller_net = {'a'}
# reports = {'a': ['b'], 'b': ['c'], 'c': ['d']}
# bids = {'a': 0, 'b': 0, 'c': 0, 'd': 1}
# print(auction(3, seller_net, reports, bids))

seller_net = {'a', 'c'}
reports = {'a': ['b', 'd'], }
bids = {'a': 20, 'b': 100, 'c': 5, 'd': 10}
print(auction(3, seller_net, reports, bids))
