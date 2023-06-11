import copy
import random
from collections import deque
import networkx as nx
from Bidder import Bidder
#from utils import create_graph_from_txt
import numpy as np


class CriticalTree:
 
    def __init__(self, Nopt, P, root):
        # bidder dell'allocazione ottimale
        self.Nopt = Nopt
        # padri critici dei bidder dell'allocazione ottimale
        self.P = P
        self.root = root
        self.tree = nx.DiGraph()
        # pesi di ogni nodo dell'albero
        self.W = dict()

    def create(self):

        # creazione dell'albero
        for n in self.Nopt:
            parents = self.P[n]
            chain = [n] + parents
            if len(chain) > 1:
                for i in range(1, len(chain)):
                    self.tree.add_edge(chain[i], chain[i - 1])
            self.tree.add_edge(self.root, chain[-1])

        # inizializzazione dei pesi

        for n in list(self.tree.nodes()):
            if n != self.root:
                self.W[n] = 0

        for n in self.Nopt:
            while n != self.root:
                self.W[n] += 1
                n = self.predecessor(n)

    def update_weights(self, winner, C):
        childs = C[winner]

        if childs:

            # individuazione dei bidder che sono figli critici del vincitore e che appartengo all'insieme
            # dell'allocazione efficiente
            bidders = set(childs).intersection(set(self.Nopt))

            if len(bidders) > 0:
                # trova il bidder con la più bassa valutazione e con un peso maggiore di zero, in quanto non
                # possiamo ottenere un elemento da un nodo che non ha un peso superiore a zero
                min_val = float('inf')
                bidder = None
                for b in bidders:
                    if b.bid < min_val and self.W[b] > 0:
                        min_val = b.bid
                        bidder = b

                if bidder is None:
                    return
                # risalgo la catena fino alla radice decrementando il peso di ogni nodo attraversato
                bidder_copy = copy.deepcopy(bidder)
                while bidder != winner:
                    self.W[bidder] -= 1
                    bidder = self.predecessor(bidder)
                # per risolvere il problema dei bidder che appartengono all'insieme Nopt e sono padri critici di
                # altri nodi appartenenti all'insieme Nopt rilanciamo l'algoritmo, in maniera tale che il padre che
                # ha perso un elemento a sua volta lo ottiene dal bidder con la più bassa valutazione nel
                # sotto-albero che ha come radice il padre
                self.update_weights(bidder_copy, C)

    def predecessor(self, node):
        return list(self.tree.predecessors(node))[0]

    def __getitem__(self, bidder):
        return list(self.tree[bidder])

    def get_childs(self, node):
        return list(self.tree.successors(node))

    def sum_weights(self, nodes):
        tot = 0
        for n in nodes:
            tot += self.W[n]
        return tot

    def nodes(self):
        return list(self.tree.nodes())


def generalize_diffusion_mechanism(k, seller_net, reports: dict, bids: dict):
    allocations = {n: False for n in bids.keys()}
    payments = {n: 0. for n in bids.keys()}

    G = nx.Graph()
    seller = Bidder('seller')

    # costruzione del grafo
    for bidder in seller_net:
        G.add_edge(seller, Bidder(bidder, bids[bidder]))

    for key in reports.keys():
        bi = Bidder(key, bids[key])
        for value in reports[key]:
            bj = Bidder(value, bids[value])
            G.add_edge(bi, bj)

    # social welfare dell'allocazione efficiente
    _, Nopt = efficient_allocation(k, list(bids.keys()), bids)
    N = {n for n in G.nodes() if n != seller}
    # dizionario contenente per ogni nodi i padri critici(nodo->[padri critici])
    P = critical_sequence(G, seller)
    # dizionario contenente per ogni nodi i figli critici (nodo->[figli critici])
    C = critical_childs_from(P)
    # lista dei vincitori
    W = []
    # inizializzazione del criticalTree
    tree = CriticalTree(Nopt, P, seller)
    tree.create()
    #print(tree[seller])
    Q = deque(tree[seller])
    Pw = []  # padri critici dei bidder vincenti
    while len(Q) > 0:
        i = Q.pop()
        # verifica di allocazione
        Ck = compute_Ck(C, P, i, k)
        if len(Ck) != 0:
            N_Ck = list(N - Ck)
        else:
            N_Ck = list(N)
        N_Ck.sort(reverse=True)
        N_Ck = N_Ck[:k]

        # se la valutazione del bidder è compresa tra le N_CK valutazioni
        # (ci possono essere più bidder che hanno la stessa valutazione)
        max_val = max(N_Ck).bid
        min_val = min(N_Ck).bid

        if min_val <= i.bid <= max_val:

            # if i in N_Ck:
            # bidder vincente
            W.append(i)
            # aggiornamento dei padri critici
            Pw = Pw + P[i]
            allocations[i.name] = True
            # calcolo del prezzo
            Cw = set([i] + C[i])
            Dw = list(N - Cw)
            Dw.sort(reverse=True)
            Dw = Dw[:k]
            sw_Dw = 0
            for bidder in Dw:
                sw_Dw += bidder.bid
            sw_Cw = 0
            for bidder in N_Ck:
                sw_Cw += bidder.bid
            # prezzo pagato dal vincitore SW_Di - (SW_Ci^k - vi)
            #print(f'payment {i.name}  - {sw_Dw - sw_Cw + i.bid}')
            payments[i.name] = sw_Dw - sw_Cw + i.bid

            childs = tree.get_childs(i)
            wchilds = 0

            if childs:
                # somma dei pesi dei figli del nodo 'i'
                wchilds = tree.sum_weights(childs)

            # in questo caso il nodo ottiene l'elemento da un suo figlio critico
            if tree.W[i] - 1 != wchilds:
                tree.update_weights(i, C)

        # aggiorno la lista Q con i figli del bidder 'i' con peso superiore a zero
        for n in tree[i]:
            if tree.W[n] > 0:
                Q.append(n)

    # calcolo delle ricompense

    # insieme dei bidder che devono ricevere la ricompensa
    # la ricompensa la ottengono i padri critici dei vincitori appartenenti al criticalTree
    br = (set(Pw) - set(W)).intersection(set(tree.nodes()))

    for i in br:
        # calcolo di SW_Di
        Cw = C[i]
        Cw = set([i] + Cw)
        Dw = list(N - Cw)
        Dw.sort(reverse=True)
        Dw = Dw[:k]
        sw_Dw = 0
        for bidder in Dw:
            sw_Dw += bidder.bid

        # calcolo di SW_Ci^k
        Ck = compute_Ck(C, P, i, k)
        N_Ck = list(N - Ck)
        N_Ck.sort(reverse=True)
        N_CK = N_Ck[:k]
        sw_ck = 0
        for n_ck in N_CK:
            sw_ck += n_ck.bid

        # SW_Di - SW_Ci^k
        reward = sw_Dw - sw_ck
        payments[i.name] = reward

    #print('winners: ', W)
    return allocations, payments


def GIDM(k, seller_net, reports: dict, bids: dict):
    allocations = {n: False for n in bids.keys()}
    payments = {n: 0. for n in bids.keys()}

    G = nx.Graph()
    seller = Bidder('seller')

    # costruzione del grafo
    for bidder in seller_net:
        G.add_edge(seller, Bidder(bidder, bids[bidder]))

    for key in reports.keys():
        bi = Bidder(key, bids[key])
        for value in reports[key]:
            bj = Bidder(value, bids[value])
            G.add_edge(bi, bj)

    # social welfare dell'allocazione efficiente
    _, Nopt = efficient_allocation(k, list(bids.keys()), bids)
    N = {n for n in G.nodes() if n != seller}
    # dizionario contenente per ogni nodi i padri critici(nodo->[padri critici])
    P = critical_sequence(G, seller)
    # dizionario contenente per ogni nodi i figli critici (nodo->[figli critici])
    C = critical_childs_from(P)
    # lista dei vincitori
    W = []
    # inizializzazione del criticalTree
    tree = CriticalTree(Nopt, P, seller)
    tree.create()
    #print(tree[seller])
    Q = deque(tree[seller])
    Pw = []  # padri critici dei bidder vincenti
    while len(Q) > 0:
        i = Q.pop()
        Ck = compute_Ck(C, P, i, k)
        Di = {i}.union(C[i])
        N_Di = N - Di
        N_Ck = N - Ck
        sw_Di,ddd = efficient_allocation(k, [b.name for b in N_Di], bids)
        sw_Ck,alloc_N_Ck = efficient_allocation(k, [b.name for b in N_Ck], bids)

        
        if i in Nopt:
            # bidder vincente
            W.append(i)
            allocations[i.name] = True
            payments[i.name] = sw_Di - sw_Ck + bids[i.name]
        else:
            if i in alloc_N_Ck:
                tree.update_weights(i, C)
                allocations[i.name] = True
                payments[i.name] = sw_Di - sw_Ck + bids[i.name]
            else:
                allocations[i.name] = False
                payments[i.name] = sw_Di - sw_Ck

        # aggiorno la lista Q con i figli del bidder 'i' con peso superiore a zero
        for n in tree[i]:
            if tree.W[n] > 0:
                Q.append(n)

    return allocations, payments


def compute_Ck(C, P, i, k):
    Ck = C[i]
    Ck.sort(reverse=True)
    # se il numero dei figli critici non supera k allora Ci^k = Ci
    # altrimenti bisogna considerare anche i genitori dei figli critici (i loro genitori che sono anche figli di b)
    if len(Ck) >= k:
        Ck = Ck[:k]
        Ck = set(Ck)
        PC = set()
        CPC = set()

        for ck in Ck:
            PC = PC.union(set(P[ck]).intersection(C[i]))

        for pc in PC:
            CPC = CPC.union(set(C[pc]).intersection(C[i]))
        Ck = Ck.union(PC).union(CPC)
    return set(Ck)


def efficient_allocation(k, bidders: list, bids: dict):
    bidder_list = [Bidder(bidder, bids[bidder]) for bidder in bidders]
    bidder_list.sort(reverse=True)
    winners = bidder_list[:k]
    sw = 0.
    for w in winners:
        sw += w.bid
    return sw, winners


def critical_sequence(G: nx.Graph, seller):
    # il seller non può essere un punto di articolazione
    articulation_points = [n for n in nx.articulation_points(G) if n != seller]
    nodes = {n for n in G.nodes() if n != seller}
    C = dict()
    empty = len(articulation_points) == 0
    for n in nodes:
        C[n] = []

    for n in articulation_points:
        reachable = set(information_diffusion(G, n, seller))
        unreachable = nodes - reachable
        unreachable.remove(n)
        for u in unreachable:
            C[u].append(n)

    if not empty:
        C = sort_critical_sequence(C, G, seller)

    return C


def information_diffusion(G: nx.Graph, bidder, seller):
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
    return visited


def critical_childs_from(P: dict):
    C = {key: [] for key in P.keys()}
    for key, values in P.items():
        for val in values:
            C[val].append(key)
    return C


def sort_critical_sequence(C: dict, G: nx.Graph, root):
    C_new = {}
    depth = depth_dict(G, root)
    for key, values in C.items():
        new_values = []
        for val in values:
            new_values.append((depth[val], val))
        # ordinamento dei nodi per profondità
        new_values.sort(key=lambda tup: tup[0], reverse=True)
        C_new[key] = [val for _, val in new_values]
    return C_new


def depth_dict(G, root):
    level = deque([root])
    visited = [root]
    depth = {root: 0}

    while len(level) > 0:

        n = level.popleft()
        for c in G[n]:
            if c not in visited:
                depth[c] = depth[n] + 1
                level.append(c)
                visited.append(c)

    return depth


if __name__ == '__main__':
    # G = nx.Graph()
    # G.add_edge('s', 'a')
    # G.add_edge('s', 'b')
    # G.add_edge('s', 'c')
    # G.add_edge('a', 'd')
    # G.add_edge('b', 'd')
    # G.add_edge('b', 'c')
    # G.add_edge('c', 'e')
    # G.add_edge('c', 'f')
    # G.add_edge('c', 'g')
    # G.add_edge('f', 'g')
    # G.add_edge('f', 'l')
    # G.add_edge('e', 'k')
    # G.add_edge('f', 'k')
    # G.add_edge('k', 'y')
    # G.add_edge('y', 'p')
    # G.add_edge('y', 'q')
    # G.add_edge('p', 'q')
    # G.add_edge('d', 'h')
    # G.add_edge('h', 'i')
    # G.add_edge('i', 'j')
    # G.add_edge('d', 'j')
    # G.add_edge('i', 'm')
    # G.add_edge('m', 'o')
    # P = critical_sequence(G, 's')
    seller_net = ['a', 'b', 'c']
    reports = {'a': ['d'], 'b': ['c', 'd'], 'c': ['e', 'f', 'g'], 'f': ['g', 'k', 'l'], 'e': ['k'], 'k': ['y'],
               'y': ['p', 'q'], 'p': ['q'], 'd': ['h', 'j'], 'h': ['i'], 'j': ['i'], 'i': ['m'], 'm': ['o']}
    bids = {'a': 7, 'b': 4, 'c': 2, 'd': 14, 'e': 8, 'f': 9, 'g': 17, 'k': 19, 'l': 10, 'y': 20, 'p': 11, 'q': 1,
            'h': 16, 'i': 6, 'j': 5, 'm': 15, 'o': 3}
    #alloc, pay = generalize_diffusion_mechanism(5, seller_net, reports, bids)
    #print(alloc, pay)
    # net4 = create_graph_from_txt('net_4', sep=' ')
    # seller = random.choice(list(net4.nodes()))
    # seller_net = []
    # for n in net4[seller]:
    #     seller_net.append(n)
    #
    # reports = {n: [] for n in net4.nodes()}
    # bids = {}
    # for n in net4.nodes():
    #     bids[n] = random.randint(10, 1000)
    #     for v in net4[n]:
    #         reports[n].append(v)
    #
    # generalize_diffusion_mechanism(10, seller_net, bids, reports)

    print("Esempio MUDAN")

    seller_net = ['a', 'b']
    reports = {'b': ['c'], 'c': ['d', 'e'], 'e': ['f'], 'f': ['g']}
    bids = {'a': 3, 'b': 1, 'c': 1, 'd': 6, 'e': 4, 'f': 7, 'g': 5}
    alloc, pay = generalize_diffusion_mechanism(4, seller_net, reports, bids)
    for i in alloc.keys():
        if alloc[i]:
            print(i)
    for i in pay.keys():
        if pay[i]!=0:
            print(i,pay[i])

    print("My version: ")

    alloc, pay = GIDM(4, seller_net, reports, bids)
    for i in alloc.keys():
        if alloc[i]:
            print(i)
    for i in pay.keys():
        if pay[i]!=0:
            print(i,pay[i])

    print("Esempio GIDM")
    seller_net = ['a', 'b', 'c']
    reports = {'a': ['d'], 'b': ['c', 'd'], 'c': ['e', 'f', 'g'], 'f': ['g', 'k', 'l'], 'e': ['k'], 'k': ['y'],
               'y': ['p', 'q'], 'p': ['q'], 'd': ['h', 'j'], 'h': ['i'], 'j': ['i'], 'i': ['m'], 'm': ['o']}
    bids = {'a': 7, 'b': 4, 'c': 2, 'd': 14, 'e': 8, 'f': 9, 'g': 17, 'k': 19, 'l': 10, 'y': 20, 'p': 11, 'q': 1,
            'h': 16, 'i': 6, 'j': 5, 'm': 15, 'o': 3}
    alloc, pay = generalize_diffusion_mechanism(5, seller_net, reports, bids)
    for i in alloc.keys():
        if alloc[i]:
            print(i)
    for i in pay.keys():
        if pay[i]!=0:
            print(i,pay[i])

    print("My version: ")

    alloc, pay = GIDM(5, seller_net, reports, bids)
    for i in alloc.keys():
        if alloc[i]:
            print(i)
    for i in pay.keys():
        if pay[i]!=0:
            print(i,pay[i])

    

