# SocialNetworkAnalysis

## Task1: Social Network Mining
Implement, if necessary, optimized versions of the social network mining algorithms seen during the course (diameter, triangles computation, clustering) 

## Task2: Centrality Measures
Implement the shapley-closeness centrality measure as defined in Michalack et al. (JAIR 2013) sec. 4.4.
Implement, if necessary, optimized versions of all studied centrality measures (degree, closeness, betweenness, PageRank, HITS-authority, HITS-hubiness, HITS-both, voterank, shapley-degree, shapley-threshold, shapley-closeness) 

## Task3: Social Network Mechanisms
Implement the VCG, the MUDAN, and the MUDAR, for selling multiple homogeneous items on a social network, with each agent only requiring a single item. The MUDAN and MUDAR algorithm are available on (Fang et al., 2023)

## Task4: Learning
Implement an environment for bandits algorithm that will model the following setting:

We are given a social network such that with a probability p(u,v) for each edge (u,v). These probabilities are unknown to the learner. The learner at each time steps interacts with this environment by choosing a vertex x. The environment set each edge (u,v) as alive with probability p(u,v), and dead otherwise. It then assigns as a reward to the learner that is equivalent to the number of nodes of the social network that are reachable from the selected vertex x through alive edges only.
