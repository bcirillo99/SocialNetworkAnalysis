import networkx as nx
from tqdm import tqdm

#VOTERANK
# The following is an extension of degree centrality, that tries to avoid to choose nodes that are too close each other
# The idea is to simulate a voting procedure on the network, in which each node votes for each neighbor on the network
# The node that takes more votes is elected and does not participate in the next elections (hence, her neighbors do not take her vote)
# Moreover, we decrease the weight of the votes issued by the neighbors of the elected node (hence, not only the elected node does not vote, but the vote of its neighbors are less influent)
# By convention, the weight of these nodes is decreased of an amount f = 1 / average degree of the network
# For more information see Zhang et al. (Nature 2016).
def voterank(G):
    rank = dict()

    n = G.number_of_nodes()

    if G.is_directed():
        # For directed graphs compute average out-degree
        avg_deg=sum(G.out_degree(v) for v in G.nodes())/n  #it computes the average degree of the network
    else:
        # For undirected graphs compute average degree
        avg_deg=sum(G.degree(v) for v in G.nodes())/n  #it computes the average degree of the network



    
    f = 1/avg_deg #it sets the decrement of weight in the network

    ability = {i:1 for i in G.nodes()} #initially the vote of each node weights 1
    for i in tqdm(range(n)):
        score = {i:0 for i in G.nodes()}

        for e in G.edges():
            # the score of a node is the sum of the vote weights of the neigbors of this node that have not been elected yet
            if e[0] not in rank.keys():
                score[e[0]] += ability[e[1]]
            if e[1] not in rank.keys() and not G.is_directed():
                score[e[1]] += ability[e[0]]

        #computes the elected vertex
        vertex = None
        maxv = -1
        for j in G.nodes():
            if j not in rank.keys() and score[j] > maxv:
                vertex = j
                maxv = score[j]

        #assigns the highest possible rank to this vertex
        rank[vertex] = n-i
        # reduces to 0 the vote weight of this vertex
        ability[vertex] = 0
        # reduces by f the vote weight of her neighbors
        for u in G[vertex]:
            ability[u] = max(0,ability[u]-f)

    return rank

