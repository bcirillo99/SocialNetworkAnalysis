# SocialNetworkAnalysis

## Task1: Social Network Mining
Implement, if necessary, optimized versions of the social network mining algorithms: diameter, triangles computation, clustering

## Task2: Centrality Measures
Implement, if necessary, optimized versions of this centrality measures: degree, closeness, betweenness, PageRank, HITS-authority, HITS-hubiness, HITS-both, voterank, shapley-degree, shapley-threshold, shapley-closeness

shapley-degree, shapley-threshold and shapley-closeness: Michalak, Tomasz P., et al. "Efficient computation of the Shapley value for game-theoretic network centrality." Journal of Artificial Intelligence Research 46 (2013): 607-650.


## Task3: Social Network Mechanisms
Implement the VCG, the MUDAN, and the MUDAR, for selling multiple homogeneous items on a social network, with each agent only requiring a single item. 

MUDAN and MUDAR: Fang, Yuan, et al. "Multi-unit Auction over a Social Network." arXiv preprint arXiv:2302.08924 (2023).

VCG: Li, Bin, et al. "Mechanism design in social networks." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 31. No. 1. 2017.

## Task4: Learning
Implement an environment for bandits algorithm that will model the following setting:

We are given a social network such that with a probability $p(u,v)$ for each edge $(u,v)$. These probabilities are unknown to the learner. The learner at each time steps interacts with this environment by choosing a vertex $x$. The environment set each edge $(u,v)$ as alive with probability $p(u,v)$, and dead otherwise. It then assigns as a reward to the learner that is equivalent to the number of nodes of the social network that are reachable from the selected vertex $x$ through alive edges only.

Given that the goal of the auctioneer is to maximize his cumulative reward, find the best bandit algorithm that the auctioneer may use in above setting (i.e., the algorithm must decide which vertex to select at each time step)