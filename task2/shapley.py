# This is a sample Python script.
import sys

from tqdm import tqdm
 
# adding Folder_2 to the system path
sys.path.insert(0, '../')
import time
from utils import *
from argparse import ArgumentParser
import networkx as nx
import random


def BFS(G, s):
  """Perform BFS of the undiscovered portion of Graph g starting at Vertex s.

  discovered is a dictionary mapping each vertex to the edge that was used to
  discover it during the BFS (s should be mapped to None prior to the call).
  Newly discovered vertices will be added to the dictionary as a result.
  """
  distances = {}
  diam = 0
  level = [s]                        # first level includes only s
  discovered = set(level)
  while len(level) > 0:
    next_level = []                  # prepare to gather newly found vertices
    for u in level:
      for v in G[u]:
        if v not in discovered:      # v is an unvisited vertex
          discovered.add(v)          # e is the tree edge that discovered v
          next_level.append(v)       # v will be further considered in next pass
          distances[v]=diam+1
    diam += 1
    level = next_level               # relabel 'next' level to become current
  return distances

# SHAPLEY DEGREE
# Compute the Shapley value for a characteristic function that extends degree centrality to coalitions.
# Specifically, the characteristic function is value(C) = |C| + |N(C)|, where N(C) is the set of nodes outside C with at least one neighbor in C.
# Even if the Shapley Value in general takes exponential time to be computed, for this particular characteristic function a polynomial time algorithm is known.
# Indeed, it has been proved that the Shapley value of node v in this case is SV[v] = 1/(1+deg(v)) + sum_{u \in N(v), u != v} 1/(1+deg(u)).
# For more information, see Michalack et al. (JAIR 2013) sec. 4.1
def shapley_degree(G):
    SV = {i:1/(1+G.in_degree(i)) if G.is_directed() else 1/(1+G.degree(i))  for i in G.nodes()}
    for u in G.nodes():
          for v in G[u]:
              if G.is_directed():
                SV[u] += 1/(1+G.in_degree(v))
              else:
                SV[u] += 1/(1+G.degree(v))
    return SV

# SHAPLEY THRESHOLD
# Consider another extension of degree centrality.
# Specifically, we assume that to influence a node outside a coalition is necessary that at least k of its neighbors are within the coalition.
# That is, the characteristic function is value(C) = |C| + |N(C,k)|, where N(C,k) is the set of nodes outside C with at least k neighbors in C.
# Even if the Shapley Value in general takes exponential time to be computed, for this particular characteristic function a polynomial time algorithm is known.
# Indeed, it has been proved that the Shapley value of node v in this case is SV[v] = min(1,k/(1+deg(v))) + sum_{u \in N(v), u != v} max(O,(deg(u)-k+1)/(deg(u)*(1+deg(u)))
# For more information, see Michalack et al. (JAIR 2013) sec. 4.2
def shapley_threshold(G, k=2):
    if G.is_directed():
      SV = {i:min(1,k/(1+G.in_degree(i))) for i in G.nodes()}
      for u in G.nodes():
          weight = max(0,(G.degree(u) - k + 1)/G.degree(u))
          #weight = max(0,(G.in_degree(u) - k + 1)/G.in_degree(u)) # dovrebbe essere in ma pure con out ci srta 0
          for v in G[u]:
              SV[u] += weight * 1/(1+G.in_degree(v))
    else:
      SV = {i:min(1,k/(1+G.degree(i))) for i in G.nodes()}
      for u in G.nodes():
          weight = max(0,(G.degree(u) - k + 1)/G.degree(u))
          for v in G[u]:
              SV[u] += weight * 1/(1+G.degree(v))
    return SV

def positive_decr_fun(x):
    return 1/(1+x)


def shapley_closeness(G, f):
    SV = {i:0 for i in G.nodes()}
    for u in tqdm(G.nodes()):
        distances_dict = BFS(G, u)
        d,w = list(distances_dict.values()),list(distances_dict.keys())
        if G.is_directed():
          index = G.in_degree(u)-1
        else:
          index = G.degree(u)-1
        sum, prev_distance, prevSV = 0, -1, -1
        if len(d) > 0:
          while index > 0:
              if d[index] == prev_distance:
                  currSV = prevSV
              else:
                  currSV = f(d[index])/(1+index)-sum
              
              SV[w[index]] += currSV
              sum += f(d[index])/(index*(1+index))
              prev_distance = d[index]
              prevSV = currSV
              index -= 1
        SV[u]+=(f(0)-sum)
    return SV