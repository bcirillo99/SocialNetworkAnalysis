import networkx as nx
import random
from collections import deque
from vcg import *
import time


def create_auction(k, num_nodes):
    G = nx.random_regular_graph(k, num_nodes)
    # scelta casuale del seller
    seller = random.choice(list(G.nodes()))
    # costruzione di seller_net
    seller_net = {bidder for bidder in G[seller]}

    # costruzione dizionario reports
    reports = {}
    level = deque([seller])
    visited = [seller]

    while len(level) > 0:

        n = level.popleft()
        for c in G[n]:
            if c not in visited:
                level.append(c)
                visited.append(c)
                if n not in reports:
                    reports[n] = [c]
                else:
                    reports[n].append(c)

    del reports[seller]
    # costruzione di bids
    bids = {}
    for n in G.nodes():
        bid = random.randrange(1, 100, 1)
        bids[n] = bid

    return seller_net, reports, bids


if __name__ == '__main__':
    seller_net, reports, bids = create_auction(2, 20)
    tic = time.time()
    print(auction(2, seller_net, reports, bids))
    toc = time.time()
    print(f'VCG time: {toc - tic}s')

