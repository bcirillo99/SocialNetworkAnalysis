import copy
import networkx as nx
import matplotlib.pyplot as plt
import ast


class Bidder:

    def __init__(self, name, bid=None):
        self.name = name
        self.bid = bid

    def __lt__(self, other):
        return self.bid < other.bid

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return '(' + self.name + ", " + str(self.bid) + ')'

    def __hash__(self):
        return hash(self.name)


def auction_mudan(k: int, seller_net: set, reports: dict, bids: dict):
    """
    Implementazione dell'algoritmo Multi-Diffusion-Auction No Reward
    :param k: intero che rappresenta il numero di elementi omogenei da vendere
    :param seller_net: bidder collegati direttamente al seller
    :param reports: dizionario che ha come chiavi i
    bidder e come valori una lista di bidder a cui riportano le informazioni dell'asta
    :param bids: dizionario che ha come chiavi differenti bidder e come valori le singole offerte
    :return: due dizionari allocation e payment. Allocation ha come chiavi i bidder e come valore un booleano, che assume
    valore True nel caso in cui il bidder ha ottenuto l'elemento altrimenti è False. Payment ha come chiavi i bidder e come
    valori il prezzo pagato dal bidder.
    """
    payments = {b: 0. for b in bids.keys()}
    allocations = {b: False for b in bids.keys()}
    G = nx.DiGraph()

    # costruzione del grafo
    seller = Bidder('seller')

    for b in seller_net:
        bidder = Bidder(b, bids[b])
        G.add_edge(seller, bidder)

    for key in reports.keys():
        bk = Bidder(key, bids[key])
        for value in reports[key]:
            bv = Bidder(value, bids[value])
            G.add_edge(bk, bv)

    """if k > len(bids.keys()):
        k = len(bids.keys())"""
    remaining_items = k

    # lista contenente i buyer esplorati esclusi i vincitori
    A_W = [bidder for bidder in G[seller]]
    # inizializzazione di P
    if len(A_W) <= remaining_items:
        P = copy.deepcopy(A_W)
    else:
        A_W.sort(reverse=True)
        P = copy.deepcopy(A_W[:remaining_items])
    W = set()
    #while len(P) > 0 and remaining_items > 0:
    i=0
    while set(P)-W: 
        # selezione del bidder vincente dall'insieme dei potenziali vincitori
        winning_bidder = choice_winner(G, P)
        P.remove(winning_bidder)
        W.add(winning_bidder)

        # settaggio del prezzo del vincitore
        pw = 0
        if remaining_items < len(A_W):
            val_bidder = A_W[remaining_items]
            pw = val_bidder.bid
        payments[winning_bidder.name] = pw
        # allocazione dell'elemento
        allocations[winning_bidder.name] = True
        remaining_items -= 1

        # esplorazione dei bidder
        bidder_level = []
        for b in A_W:
            for nb in G[b]:
                if nb not in A_W and nb not in W:
                    bidder_level.append(nb)
        A_W = A_W + bidder_level
        A_W.remove(winning_bidder)

        # aggiornamento dei potenziali vincitori
        
        if len(A_W) <= remaining_items:
            P = copy.deepcopy(A_W)
        else:
            A_W.sort(reverse=True)
            P = copy.deepcopy(A_W[:remaining_items])
        """print("len P:", len(P))
        print("len W:", len(W))"""

    return allocations, payments


def auction_mudar(k: int, seller_net: set, reports: dict, bids: dict):
    """
      Implementazione dell'algoritmo Multi-Diffusion-Auction Reward
      :param k: intero che rappresenta il numero di elementi omogenei da vendere
      :param seller_net: bidder collegati direttamente al seller
      :param reports: dizionario che ha come chiavi i
      bidder e come valori una lista di bidder a cui riportano le informazioni dell'asta
      :param bids: dizionario che ha come chiavi differenti bidder e come valori le singole offerte
      :return: due dizionari allocation e payment. Allocation ha come chiavi i bidder e come valore un booleano, che assume
      valore True nel caso in cui il bidder ha ottenuto l'elemento altrimenti è False. Payment ha come chiavi i bidder e come
      valori il prezzo pagato dal bidder. Nel caso in cui il pagamento è negativo rappresenta la ricompensa da parte del seller.
      """
    payments = {b: 0. for b in bids.keys()}
    allocations = {b: False for b in bids.keys()}
    G = nx.DiGraph()

    # costruzione del grafo
    seller = Bidder('seller')

    for b in seller_net:
        bidder = Bidder(b, bids[b])
        G.add_edge(seller, bidder)

    for key in reports.keys():
        bk = Bidder(key, bids[key])
        for value in reports[key]:
            bv = Bidder(value, bids[value])
            G.add_edge(bk, bv)

    # lista dei vincitori
    W = []
    level = [seller]
    A = []  # visited
    A_W = []  # A/W

    # lista contenente i buyer esplorati esclusi i vincitori

    level = [bidder for bidder in G[seller]]
    A_W = [bidder for bidder in G[seller]]
    A = [bidder for bidder in G[seller]]
    # inizializzazione di P
    if len(A_W) <= k:
        P = copy.deepcopy(A_W)
    else:
        A_W.sort(reverse=True)
        P = copy.deepcopy(A_W[:k])
    W = set()

    while set(P)-W:
        winning_bidder = choice_winner(G, P)
        
        # aggiornamento dell'insieme A_W
        A_W.remove(winning_bidder)
        # settaggio del prezzo
        A.sort(reverse=True)
        pw = 0
        if k < len(A):
            # a differenza di MUDAN nel calcolo del prezzo l'insieme utilizzato è A e non A_W
            pw = A[k].bid
        payments[winning_bidder.name] = pw
        W.add(winning_bidder)

        # esplorazione dei bidder
        new_level = []
        for n in level:
            for nb in G[n]:
                if nb not in A:
                    new_level.append(nb)
                    A.append(nb)
                    A_W.append(nb)
        level = new_level
        A_W.sort(reverse=True)
        # i potenziali vincitori sono i primi k + 1 buyer con la più alta valutazione
        P = A_W[:k]
        # scelta del vincitore dall'insieme dei potenziali vincitori

    # calcolo delle ricompense
    W = sorted(W,reverse=True)
    # insieme dei bidder che hanno ottenuto la ricompensa
    Wr = W[k:]
    # insieme dei bidder che hanno ottenuto l'elemento
    Wa = W[:k]
    for bidder in Wr:
        # la ricompensa è pari al'utilità
        payments[bidder.name] = payments[bidder.name] - bids[bidder.name]

    for bidder in Wa:
        allocations[bidder.name] = True
    
    return allocations, payments


def choice_winner(G, P):
    max_sigma = 0
    winning_bidder = P[0] #If all bidders have a 0 out_grade the first one is the winner
    for bidder in P:
        
        r = G.out_degree(bidder)
        if r > max_sigma:
            max_sigma = r
            winning_bidder = bidder
    
    return winning_bidder


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
    print("SSSSSS")
    """seller_net = {'1'}
    reports ={'1': ['2', '3', '4', '5', '6'], '2': ['7'], '3': ['8'], '4': ['9'], '5': ['10', '11'], '6': ['12'], '10': ['13', '14', '15'], '12': ['16'], '14': ['17'], '17': ['18'], '18': ['19'], '19': ['20']}
    bids = {'1': 21, '2': 71, '3': 10, '4': 34, '5': 62, '6': 7, '7': 32, '8': 12, '9': 52, '10': 6, '11': 96, '12': 91, '13': 86, '14': 74, '15': 81, '16': 43, '17': 82, '18': 69, '19': 18, '20': 56}
    print("MUDAR")
    allocations, payments = auction_mudar(19, seller_net, reports, bids)
    print("\npayments:")
    print(payments)
    print("\nallocation:")
    print(allocations)
    print(sum(payments.values()))
    print(auction_results(allocations, payments))

    print("MUDAN")
    allocations, payments = auction_mudan(19, seller_net, reports, bids)
    print("\npayments:")
    print(payments)
    print("\nallocation:")
    print(allocations)
    print(sum(payments.values()))
    print(auction_results(allocations, payments))"""

    """seller_net = {'1'}
    reports ={'1': ['2', '3', '4', '5', '6'], '2': ['7'], '3': ['8'], '4': ['9'], '5': ['10', '11'], '6': ['12'], '10': ['13', '14', '15'], '12': ['16'], '14': ['17'], '17': ['18'], '18': ['19'], '19': ['20']}
    bids = {'1': 21, '2': 71, '3': 10, '4': 34, '5': 62, '6': 7, '7': 32, '8': 12, '9': 52, '10': 6, '11': 96, '12': 91, '13': 86, '14': 74, '15': 81, '16': 43, '17': 82, '18': 69, '19': 18, '20': 56}
    print(bids)
    print(sorted(list(bids.values()),reverse=True))
    print("MUDAR")
    allocations, payments = auction_mudar(18, seller_net, reports, bids)
    print("\npayments:")
    print(payments)
    print("\nallocation:")
    print(allocations)
    print(sum(payments.values()))
    print(auction_results(allocations, payments))


    seller_net = {'6582'}
    reports={'6582': ['11062'], '11062': ['893']}
    bids = {'6582': 21, '893': 71, '11062': 10}
    allocations, payments = auction_mudar(1, seller_net, reports, bids)
    print("\npayments:")
    print(payments)
    print("\nallocation:")
    print(allocations)
    print(sum(payments.values()))
    print(auction_results(allocations, payments))"""
    """
    with open("data.txt", "r") as f:
        seller_net = ast.literal_eval(f.readline())
        reports = ast.literal_eval(f.readline())
        bids = ast.literal_eval(f.readline())
        k = int(f.readline())

    allocations, payments = auction_mudar(k, seller_net, reports, bids)

    print(sum(payments.values()))
    auction_results(allocations,bids, payments)

    with open("data.txt", "r") as f:
        seller_net = ast.literal_eval(f.readline())
        reports = ast.literal_eval(f.readline())
        bids = ast.literal_eval(f.readline())
        k = int(f.readline())

    allocations, payments = auction_mudan(k, seller_net, reports, bids)


    print(sum(payments.values()))
    auction_results(allocations,bids, payments)"""

    """seller_net = {'1'}
    reports ={'1': ['2', '3', '4', '5', '6'], '2': ['7'], '3': ['8'], '4': ['9'], '5': ['10', '11'], '6': ['12'], '10': ['13', '14', '15'], '12': ['16'], '14': ['17'], '17': ['18'], '18': ['19'], '19': ['20']}
    bids = {'1': 21, '2': 71, '3': 10, '4': 34, '5': 62, '6': 7, '7': 32, '8': 12, '9': 52, '10': 6, '11': 96, '12': 91, '13': 86, '14': 74, '15': 81, '16': 43, '17': 82, '18': 69, '19': 18, '20': 56}
    allocations, payments = auction_mudan(555, seller_net, reports, bids)
    print("MUDAN K: ",555)
    auction_results(allocations,bids, payments)"""

    seller_net = {'1'}
    reports ={'1': ['2', '3', '4', '5', '6'], '2': ['7'], '3': ['8'], '4': ['9'], '5': ['10', '11'], '6': ['12'], '10': ['13', '14', '15'], '12': ['16'], '14': ['17'], '17': ['18'], '18': ['19'], '19': ['20']}
    bids = {'1': 21, '2': 71, '3': 10, '4': 34, '5': 62, '6': 7, '7': 32, '8': 12, '9': 52, '10': 6, '11': 96, '12': 91, '13': 86, '14': 74, '15': 81, '16': 43, '17': 82, '18': 69, '19': 18, '20': 56}
    allocations, payments = auction_mudan(18, seller_net, reports, bids)
    print("MUDAN K: ",18)
    """print("\npayments:")
    print(payments)
    print("\nallocation:")
    print(allocations)"""
    print(sum(payments.values()))
    auction_results(allocations,bids, payments)

    seller_net = {'1'}
    reports ={'1': ['2', '3', '4', '5', '6'], '2': ['7'], '3': ['8'], '4': ['9'], '5': ['10', '11'], '6': ['12'], '10': ['13', '14', '15'], '12': ['16'], '14': ['17'], '17': ['18'], '18': ['19'], '19': ['20']}
    bids = {'1': 21, '2': 71, '3': 10, '4': 34, '5': 62, '6': 7, '7': 32, '8': 12, '9': 52, '10': 6, '11': 96, '12': 91, '13': 86, '14': 74, '15': 81, '16': 43, '17': 82, '18': 69, '19': 18, '20': 56}
    allocations, payments = auction_mudar(5, seller_net, reports, bids)
    print("MUDAR K: ",555)
    auction_results(allocations,bids, payments)
    print("\npayments:")
    print(payments)
    print("\nallocation:")
    print(allocations)

    print("\n\n")
    print({k: v for k, v in sorted(bids.items(), key=lambda item: item[1], reverse=True)})
