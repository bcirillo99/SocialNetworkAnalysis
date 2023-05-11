import copy
import networkx as nx


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

    remaining_items = k

    # lista contenente i buyer esplorati esclusi i vincitori
    A_W = [bidder for bidder in G[seller]]
    # inizializzazione di P
    if len(A_W) <= remaining_items:
        P = copy.deepcopy(A_W)
    else:
        A_W.sort(reverse=True)
        P = copy.deepcopy(A_W[:remaining_items])

    while len(P) > 0 and remaining_items > 0:

        # selezione del bidder vincente dall'insieme dei potenziali vincitori
        winning_bidder = choice_winner(G, P)
        P.remove(winning_bidder)

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
                bidder_level.append(nb)
        A_W = A_W + bidder_level
        A_W.remove(winning_bidder)

        # aggiornamento dei potenziali vincitori
        if len(A_W) <= remaining_items:
            P = copy.deepcopy(A_W)
        else:
            A_W.sort(reverse=True)
            P = copy.deepcopy(A_W[:remaining_items])

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
    while len(level) > 0:

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
        W.append(winning_bidder)

    # calcolo delle ricompense
    W.sort(reverse=True)
    # insieme dei bidder che hanno ottenuto la ricompensa
    Wr = W[k:]
    # insieme dei bidder che hanno ottenuto l'elemento
    Wa = W[:k]
    for bidder in Wr:
        # la ricompensa è pari al'utilità
        payments[bidder.name] = payments[bidder.name] - bids[bidder.name]

    for bidder in Wa:
        allocations[bidder.name] = True

    return payments, allocations


def choice_winner(G, P):
    max_sigma = 0
    winning_bidder = None
    for bidder in P:
        r = G.degree(bidder)
        if r > max_sigma:
            max_sigma = r
            winning_bidder = bidder
    return winning_bidder


def auction_results(allocations: dict, bids: dict):
    rw = 0.
    sw = 0.
    for bidder, alloc in allocations.items():
        if alloc:
            sw += bids[bidder]
        else:
            rw += bids[bidder]
    return sw, rw


if __name__ == '__main__':
    seller_net = {'a', 'b'}
    reports = {'b': ['c'], 'c': ['d', 'e'], 'e': ['f'], 'f': ['g']}
    bids = {'a': 3., 'b': 1, 'c': 1, 'd': 6, 'e': 4, 'f': 7, 'g': 5}
    allocations, payments = auction_mudan(4, seller_net, reports, bids)
    print(allocations, payments)
    print(auction_results(allocations, payments))
    print(auction_mudar(4, seller_net, reports, bids))
    # seller_net = {'a'}
    # reports = {'a': ['d', 'c']}
    # bids = {'a': 2, 'b': 3, 'c': 5, 'd': 6}
    # print(auction_mudan(4, seller_net, reports, bids))
    a = {1, 2, 3}
    b = {4, 5, 6}
