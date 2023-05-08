import time

from argparse import ArgumentParser
import networkx as nx
import random
from scipy.sparse import linalg


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

def BFS2(G, s):
  """Perform BFS of the undiscovered portion of Graph g starting at Vertex s.

  discovered is a dictionary mapping each vertex to the edge that was used to
  discover it during the BFS (s should be mapped to None prior to the call).
  Newly discovered vertices will be added to the dictionary as a result.
  """
  distances = {}
  n = len(G)
  diam=0
  clevel=[s]
  visited=set(s)
  while len(visited) < n:
      nlevel=[]
      while(len(clevel) > 0):
          c=clevel.pop()
          print(c,G[c])
          for v in G[c]:
              if v not in visited:
                  visited.add(v)
                  nlevel.append(v)
                  distances[v]=diam+1
      clevel = nlevel
      diam += 1
  return distances
def diameter(G, sample=None):
    nodes=G.nodes()
    n = len(nodes)
    diam = 0
    if sample is None:
        sample = nodes

    for u in sample:
        udiam=0
        clevel=[u]
        visited=set(u)
        while len(clevel) > 0:
            nlevel=[]
            for c in clevel:
                for v in G[c]:
                    if v not in visited:
                        visited.add(v)
                        nlevel.append(v)
            clevel = nlevel
            udiam += 1
            print(clevel,udiam,c)
        if udiam > diam:
            diam = udiam

    return diam-1
def diameter2(G, sample=None):
    nodes=G.nodes()
    n = len(nodes)
    diam = 0
    if sample is None:
        sample = nodes

    for u in sample:
        udiam=0
        clevel=[u]
        visited=set(u)
        while len(visited) < n:
            nlevel=[]
            while(len(clevel) > 0):
                c=clevel.pop()
                for v in G[c]:
                    if v not in visited:
                        visited.add(v)
                        nlevel.append(v)
            
            clevel = nlevel
            udiam += 1
            print(clevel,udiam,len(visited))
            
        if udiam > diam:
            diam = udiam

    return diam

G = nx.Graph()
G.add_edge('A', 'B')
G.add_edge('A','D')
G.add_edge('B', 'E')
G.add_edge('B', 'C')
G.add_edge('C', 'E')
G.add_edge('C', 'D')
G.add_edge('C', 'F')
#G.add_edge('C','A')


print(BFS(G, 'C').values(),BFS(G, 'C').keys())
#print(BFS2(G, 'C').values(),BFS2(G, 'C').keys())

print(diameter(G))
print(diameter2(G))