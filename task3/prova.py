import networkx as nx
from collections import deque
import random
from tqdm import tqdm
from multi_diffusion_auction import Bidder



seller_net = {'1'}
reports ={'1': ['2', '3', '4', '5', '6'], '2': ['7'], '3': ['8'], '4': ['9'], '5': ['10', '11'], '6': ['12'], '10': ['13', '14', '15'], '12': ['16'], '14': ['17'], '17': ['18'], '18': ['19'], '19': ['20']}
bids = {'1': 21, '2': 71, '3': 10, '4': 34, '5': 62, '6': 7, '7': 32, '8': 12, '9': 52, '10': 6, '11': 96, '12': 91, '13': 86, '14': 74, '15': 81, '16': 43, '17': 82, '18': 69, '19': 18, '20': 56}
print(sum(bids.values()))
G = nx.DiGraph()
seller = Bidder('seller')

for b in seller_net:
    bidder = Bidder(b, bids[b])
    G.add_edge(seller, bidder)

for key in reports.keys():
    bk = Bidder(key, bids[key])
    for value in reports[key]:
        bv = Bidder(value, bids[value])
        G.add_edge(bk, bv)


bidder = Bidder('5','62')
level = deque([seller])
visited = [seller]

while len(level) > 0:

    n = level.popleft()
    for c in G[n]:
        print(c,c != bidder)
        if (c not in visited) and (c != bidder):
            level.append(c)
            visited.append(c)

print(visited)