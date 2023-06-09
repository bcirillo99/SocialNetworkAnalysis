import math
import networkx as nx
import random

import numpy as np

class SocNetMec_UCB:
    
    def __init__(self, G, T, k, auctions, arms_set, auction):
        self.G = G
        self.T = T
        self.k = k
        self.auctions = auctions
        self._type_auction = auction

        self.__arms_set = arms_set #initialize the set of arms
        self.__t = 0
        # Next initialization will serve for computing the best strategy during exploitation steps
        self.__num = {a:0 for a in self.__arms_set} #It saves the number of times arm a has been selected
        self.__rew = {a:0 for a in self.__arms_set} #It saves the cumulative reward achieved by arm a when selected
        self.__avgrew = {a:0 for a in self.__arms_set}  #It saves the average reward achieved by arm a until the current time step

        #It saves the ucb value of each arm until the current time step
        #It is initialised to infinity in order to allow that each arm is selected at least once
        self.__ucb = {a:float('inf') for a in self.__arms_set}
        self.__n_nodes = []

    def get_n(self):
        return self.__n_nodes
    
    def __init(self, t):
        self.__t = t
        a_t = max(self.__ucb, key=self.__ucb.get)  #We choose the arm that has the highest average revenue
        auction_t = self.auctions[self._type_auction]
        return a_t, auction_t

    def __invite(self, t, u, v, auction, prob, val):
        if prob(u,v):
          bv = val(t,v)
          sv = self.G[v]
          if not auction[0]:
            bv = random.uniform(bv*2/3, bv)
          if not auction[1]:
            new_sv = set()
            for n in sv:
                r = random.random()
                if r <= 0.15:
                    new_sv.add(n) 
            sv = new_sv
          return bv, sv
        else:
          return False, False
        
    def get_best_arm_approx(self):
        return self.__avgrew[max(self.__avgrew, key=self.__avgrew.get)]
        
    def __receive_reward(self, arm, auction, prob, val):
        """
        It select the reward for the given arm equal to the number of nodes in the graph
        that are reachable from the selected vertex x (the arm) only through alive edges

        Parameters
        ----------
        arm:
            the arm (vertex) that the learner want to use

        Returns
        ---------
            the reward assigned by the environment
        """
        clevel=[arm]
        visited=set(arm)
        invited=set(self.G[arm])
        seller_net = set()
        reports = dict()
        bids = dict()
        while len(clevel) > 0:
            nlevel=[]
            for c in clevel:
                for v in self.G[c]:
                    if v not in visited and v in invited:
                        bv,sv = self.__invite(self.__t-1,c,v,auction, prob, val)
                        #print("invited from ",v,": ",sv)
                        if bv:
                          visited.add(v)
                          nlevel.append(v)
                          for n in sv:
                            if n != arm:
                                invited.add(n)
                          bids[v] = bv
                          if c == arm:
                              seller_net.add(v)
                          else:
                              if c in reports:
                                  reports[c].append(v)
                              else:
                                  reports[c] = [v]
            clevel = nlevel
        """print("seller net:")
        print(seller_net)
        print("reports:")
        print(reports)
        print("bids:")
        print(bids)"""
        if len(seller_net)>0:
            """G1 = nx.DiGraph()

            # costruzione del grafo
            for bidder in seller_net:
                G1.add_edge('seller', bidder)

            for key in reports.keys():
                for value in reports[key]:
                    G1.add_edge(key, value)
            n_nodes = G1.number_of_nodes()-1"""
            allocation, payments = auction[2](self.k, seller_net, reports, bids)
            reward = sum(payments.values())
        else:
            #n_nodes = 0
            reward = 0
        #self.__n_nodes.append(n_nodes)
        """print("payments:")
        print(payments)
        print("allocation:")
        print(allocation)"""
        return reward
    

    def run(self, t, prob, val):
        t+=1
        a_t, auction_t = self.__init(t)
        ##print("Selected arm at time ", t, " : ",a_t)
        reward = self.__receive_reward(a_t, auction_t, prob, val) #We save the reward assigned by the environment
        ##print("recived reward: ", reward)
        # We update the number of times arm a_t has been chosen, its cumulative reward and its UCB value
        self.__num[a_t] += 1
        self.__rew[a_t] += reward
        self.__avgrew[a_t] = self.__rew[a_t]/self.__num[a_t]
        self.__ucb[a_t] = self.__avgrew[a_t] + math.sqrt(2*math.log(self.__t)/self.__num[a_t])

        return reward
    
class SocNetMec_TH:
    
    def __init__(self, G, T, k, auctions, arms_set, auction, gaussian_dist=None):
        self.G = G
        self.T = T
        self.k = k
        self.auctions = auctions
        self._type_auction = auction

        self.__arms_set = arms_set #initialize the set of arms
        self.__t = 0
        # Next initialization will serve for computing the best strategy during exploitation steps
        self.__num = {a:0 for a in self.__arms_set} #It saves the number of times arm a has been selected
        self.__rew = {a:0 for a in self.__arms_set} #It saves the cumulative reward achieved by arm a when selected
        self.__avgrew = {a:0 for a in self.__arms_set}  #It saves the average reward achieved by arm a until the current time step

        #It saves the ucb value of each arm until the current time step
        #It is initialised to infinity in order to allow that each arm is selected at least once
        if gaussian_dist is not None:
            self.__gaussian_dist = gaussian_dist
        else:
            self.__gaussian_dist = {a:(0,1) for a in arms_set}  


    def __init(self, t):
        self.__t = t
        max = 0
        a_t = None
        for arm in self.__arms_set:
            mu, sigma = self.__gaussian_dist[arm]
            gauss_value = np.random.normal(mu, sigma)
            if gauss_value > max:
                max = gauss_value
                a_t = arm
        auction_t = self.auctions[self._type_auction]
        if a_t is None:
            a_t = random.choice(self.__arms_set)
        return a_t, auction_t

    def __invite(self, t, u, v, auction, prob, val):
        if prob(u,v):
          bv = val(t,v)
          sv = self.G[v]
          if not auction[0]:
            bv = random.uniform(bv*2/3, bv)
          if not auction[1]:
            new_sv = set()
            for n in sv:
                r = random.random()
                if r <= 0.15:
                    new_sv.add(n) 
            sv = new_sv
          return bv, sv
        else:
          return False, False
        
    def get_best_arm_approx(self):
        return self.__avgrew[max(self.__avgrew, key=self.__avgrew.get)]
        
    def __receive_reward(self, arm, auction, prob, val):
        """
        It select the reward for the given arm equal to the number of nodes in the graph
        that are reachable from the selected vertex x (the arm) only through alive edges

        Parameters
        ----------
        arm:
            the arm (vertex) that the learner want to use

        Returns
        ---------
            the reward assigned by the environment
        """
        clevel=[arm]
        visited=set(arm)
        invited=set(self.G[arm])
        seller_net = set()
        reports = dict()
        bids = dict()
        while len(clevel) > 0:
            nlevel=[]
            for c in clevel:
                for v in self.G[c]:
                    if v not in visited and v in invited:
                        bv,sv = self.__invite(self.__t,c,v,auction, prob, val)
                        #print("invited from ",v,": ",sv)
                        if bv:
                          visited.add(v)
                          nlevel.append(v)
                          for n in sv:
                            if n != arm:
                                invited.add(n)
                          bids[v] = bv
                          if c == arm:
                              seller_net.add(v)
                          else:
                              if c in reports:
                                  reports[c].append(v)
                              else:
                                  reports[c] = [v]
            clevel = nlevel
        """print("seller net:")
        print(seller_net)
        print("reports:")
        print(reports)
        print("bids:")
        print(bids)"""
        if len(seller_net)>0:
            allocation, payments = auction[2](self.k, seller_net, reports, bids)
            reward = sum(payments.values())
        else:
            reward = 0
        """print("payments:")
        print(payments)
        print("allocation:")
        print(allocation)"""
        return reward
    

    def run(self, t, prob, val):
        a_t, auction_t = self.__init(t)
        #print("Selected arm at time ", t, " : ",a_t)
        reward = self.__receive_reward(a_t, auction_t, prob, val) #We save the reward assigned by the environment
        #print("recived reward: ", reward)
        # We update the number of times arm a_t has been chosen, its cumulative reward and its UCB value
        self.__num[a_t] += 1
        self.__rew[a_t] += reward
        self.__avgrew[a_t] = self.__rew[a_t]/self.__num[a_t]

        mu, sigma = self.__gaussian_dist[a_t]

        n = self.__num[a_t]
        #mu = (mu * (n-1) + reward) / n
        mu = self.__avgrew[a_t]
        sigma = (sigma * (n - 2) + (reward - mu) ** 2) / (n - 1) if n > 1 else 1
        self.__gaussian_dist[a_t] =  (mu, sigma)
        return reward


class SocNetMec_EPS:
    
    def __init__(self, G, T, k, auctions, arms_set, auction,eps):
        self.G = G
        self.T = T
        self.k = k
        self.auctions = auctions
        self._type_auction = auction

        self.__eps = eps
        self.__arms_set = arms_set #initialize the set of arms
        self.__t = 0
        # Next initialization will serve for computing the best strategy during exploitation steps
        self.__num = {a:0 for a in self.__arms_set} #It saves the number of times arm a has been selected
        self.__rew = {a:0 for a in self.__arms_set} #It saves the cumulative reward achieved by arm a when selected
        self.__avgrew = {a:0 for a in self.__arms_set}  #It saves the average reward achieved by arm a until the current time step

        #It saves the ucb value of each arm until the current time step
        #It is initialised to infinity in order to allow that each arm is selected at least once

    def __init(self, t):
        self.__t = t
        r = random.random()
        if r <= self.__eps(self.__t):
            a_t = random.choice(self.__arms_set) #We choose an arm uniformly at random
        else:
            a_t = max(self.__avgrew, key=self.__avgrew.get) #We choose the arm that has the highest average revenue
        auction_t = self.auctions[self._type_auction]
        return a_t, auction_t

    def __invite(self, t, u, v, auction, prob, val):
        if prob(u,v):
          bv = val(t,v)
          sv = self.G[v]
          if not auction[0]:
            bv = random.uniform(bv*2/3, bv)
          if not auction[1]:
            new_sv = set()
            for n in sv:
                r = random.random()
                if r <= 0.15:
                    new_sv.add(n) 
            sv = new_sv
          return bv, sv
        else:
          return False, False
        
    def get_best_arm_approx(self):
        return self.__avgrew[max(self.__avgrew, key=self.__avgrew.get)]
        
    def __receive_reward(self, arm, auction, prob, val):
        """
        It select the reward for the given arm equal to the number of nodes in the graph
        that are reachable from the selected vertex x (the arm) only through alive edges

        Parameters
        ----------
        arm:
            the arm (vertex) that the learner want to use

        Returns
        ---------
            the reward assigned by the environment
        """
        clevel=[arm]
        visited=set(arm)
        invited=set(self.G[arm])
        seller_net = set()
        reports = dict()
        bids = dict()
        while len(clevel) > 0:
            nlevel=[]
            for c in clevel:
                for v in self.G[c]:
                    if v not in visited and v in invited:
                        bv,sv = self.__invite(self.__t,c,v,auction, prob, val)
                        #print("invited from ",v,": ",sv)
                        if bv:
                          visited.add(v)
                          nlevel.append(v)
                          for n in sv:
                            if n != arm:
                                invited.add(n)
                          bids[v] = bv
                          if c == arm:
                              seller_net.add(v)
                          else:
                              if c in reports:
                                  reports[c].append(v)
                              else:
                                  reports[c] = [v]
            clevel = nlevel
        """print("seller net:")
        print(seller_net)
        print("reports:")
        print(reports)
        print("bids:")
        print(bids)"""
        if len(seller_net)>0:
            allocation, payments = auction[2](self.k, seller_net, reports, bids)
            reward = sum(payments.values())
        else:
            reward = 0
        """print("payments:")
        print(payments)
        print("allocation:")
        print(allocation)"""
        return reward
    

    def run(self, t, prob, val):
        a_t, auction_t = self.__init(t)
        #print("Selected arm at time ", t, " : ",a_t)
        reward = self.__receive_reward(a_t, auction_t, prob, val) #We save the reward assigned by the environment
        #print("recived reward: ", reward)
        # We update the number of times arm a_t has been chosen, its cumulative reward and its UCB value
        self.__num[a_t] += 1
        self.__rew[a_t] += reward
        self.__avgrew[a_t] = self.__rew[a_t]/self.__num[a_t]

        return reward
    
class SocNetMec_UCB_mixed:
    
    def __init__(self, G, T, k, auctions, arms_set):
        self.G = G
        self.T = T
        self.k = k
        self.auctions = auctions

        self.__arms_set = arms_set #initialize the set of arms
        self.__t = 0
        # Next initialization will serve for computing the best strategy during exploitation steps
        self.__num = {a[0]:0 for a in self.__arms_set} #It saves the number of times arm a has been selected
        self.__rew = {a[0]:0 for a in self.__arms_set} #It saves the cumulative reward achieved by arm a when selected
        self.__avgrew = {a[0]:0 for a in self.__arms_set}  #It saves the average reward achieved by arm a until the current time step

        #It saves the ucb value of each arm until the current time step
        #It is initialised to infinity in order to allow that each arm is selected at least once
        self.__ucb = {a:float('inf') for a in self.__arms_set}
        #self.__n_nodes = []

    """def get_n(self):
        return self.__n_nodes"""
    def __init(self, t):
        self.__t = t
        a_t,auction = max(self.__ucb, key=self.__ucb.get)  #We choose the arm that has the highest average revenue
        auction_t = self.auctions[auction]
        return a_t, auction_t

    def __invite(self, t, u, v, auction, prob, val):
        if prob(u,v):
          bv = val(t,v)
          sv = self.G[v]
          if not auction[0]:
            bv = random.uniform(bv*2/3, bv)
          if not auction[1]:
            new_sv = set()
            for n in sv:
                r = random.random()
                if r <= 0.15:
                    new_sv.add(n) 
            sv = new_sv
          return bv, sv
        else:
          return False, False
        
    def get_best_arm_approx(self):
        return self.__avgrew[max(self.__avgrew, key=self.__avgrew.get)]
        
    def __receive_reward(self, arm, auction, prob, val):
        """
        It select the reward for the given arm equal to the number of nodes in the graph
        that are reachable from the selected vertex x (the arm) only through alive edges

        Parameters
        ----------
        arm:
            the arm (vertex) that the learner want to use

        Returns
        ---------
            the reward assigned by the environment
        """
        clevel=[arm]
        visited=set(arm)
        invited=set(self.G[arm])
        seller_net = set()
        reports = dict()
        bids = dict()
        while len(clevel) > 0:
            nlevel=[]
            for c in clevel:
                for v in self.G[c]:
                    if v not in visited and v in invited:
                        bv,sv = self.__invite(self.__t-1,c,v,auction, prob, val)
                        #print("invited from ",v,": ",sv)
                        if bv:
                          visited.add(v)
                          nlevel.append(v)
                          for n in sv:
                            if n != arm:
                                invited.add(n)
                          bids[v] = bv
                          if c == arm:
                              seller_net.add(v)
                          else:
                              if c in reports:
                                  reports[c].append(v)
                              else:
                                  reports[c] = [v]
            clevel = nlevel
        """print("seller net:")
        print(seller_net)
        print("reports:")
        print(reports)
        print("bids:")
        print(bids)"""
        if len(seller_net)>0:
            """G1 = nx.DiGraph()

            # costruzione del grafo
            for bidder in seller_net:
                G1.add_edge('seller', bidder)

            for key in reports.keys():
                for value in reports[key]:
                    G1.add_edge(key, value)
            n_nodes = G1.number_of_nodes()-1"""
            allocation, payments = auction[2](self.k, seller_net, reports, bids)
            reward = sum(payments.values())
        else:
            #n_nodes = 0
            reward = 0
        #self.__n_nodes.append(n_nodes)
        """print("payments:")
        print(payments)
        print("allocation:")
        print(allocation)"""
        return reward
    

    def run(self, t, prob, val):
        t+=1
        a_t, auction_t = self.__init(t)
        #print("Selected arm at time ", t, " : ",a_t)
        reward = self.__receive_reward(a_t, auction_t, prob, val) #We save the reward assigned by the environment
        #print("recived reward: ", reward)
        # We update the number of times arm a_t has been chosen, its cumulative reward and its UCB value
        self.__num[a_t] += 1
        self.__rew[a_t] += reward
        self.__avgrew[a_t] = self.__rew[a_t]/self.__num[a_t]
        self.__ucb[a_t] = self.__avgrew[a_t] + math.sqrt(2*math.log(self.__t)/self.__num[a_t])

        return reward
    

class SocNetMec_Bayesian_UCB:
    
    def __init__(self, G, T, k, auctions, arms_set, auction, gaussian_dist=None):
        self.G = G
        self.T = T
        self.k = k
        self.auctions = auctions
        self._type_auction = auction

        self.__arms_set = arms_set #initialize the set of arms
        self.__t = 0
        # Next initialization will serve for computing the best strategy during exploitation steps
        self.__num = {a:0 for a in self.__arms_set} #It saves the number of times arm a has been selected
        self.__rew = {a:0 for a in self.__arms_set} #It saves the cumulative reward achieved by arm a when selected
        self.__avgrew = {a:0 for a in self.__arms_set}  #It saves the average reward achieved by arm a until the current time step

        #It saves the ucb value of each arm until the current time step
        #It is initialised to infinity in order to allow that each arm is selected at least once
        self.__ucb = {a:float('inf') for a in self.__arms_set}
        self.__ucb_scale = 1.96
        #self.__n_nodes = []

        if gaussian_dist is not None:
            self.__gaussian_dist = gaussian_dist
        else:
            self.__gaussian_dist = {a:(0,1) for a in arms_set}  

    """def get_n(self):
        return self.__n_nodes"""
    def __init(self, t):
        self.__t = t
        a_t = max(self.__ucb, key=self.__ucb.get)  #We choose the arm that has the highest average revenue
        auction_t = self.auctions[self._type_auction]
        return a_t, auction_t

    def __invite(self, t, u, v, auction, prob, val):
        if prob(u,v):
          bv = val(t,v)
          sv = self.G[v]
          if not auction[0]:
            bv = random.uniform(bv*2/3, bv)
          if not auction[1]:
            new_sv = set()
            for n in sv:
                r = random.random()
                if r <= 0.15:
                    new_sv.add(n) 
            sv = new_sv
          return bv, sv
        else:
          return False, False
        
    def get_best_arm_approx(self):
        return self.__avgrew[max(self.__avgrew, key=self.__avgrew.get)]
        
    def __receive_reward(self, arm, auction, prob, val):
        """
        It select the reward for the given arm equal to the number of nodes in the graph
        that are reachable from the selected vertex x (the arm) only through alive edges

        Parameters
        ----------
        arm:
            the arm (vertex) that the learner want to use

        Returns
        ---------
            the reward assigned by the environment
        """
        clevel=[arm]
        visited=set(arm)
        invited=set(self.G[arm])
        seller_net = set()
        reports = dict()
        bids = dict()
        while len(clevel) > 0:
            nlevel=[]
            for c in clevel:
                for v in self.G[c]:
                    if v not in visited and v in invited:
                        bv,sv = self.__invite(self.__t-1,c,v,auction, prob, val)
                        #print("invited from ",v,": ",sv)
                        if bv:
                          visited.add(v)
                          nlevel.append(v)
                          for n in sv:
                            if n != arm:
                                invited.add(n)
                          bids[v] = bv
                          if c == arm:
                              seller_net.add(v)
                          else:
                              if c in reports:
                                  reports[c].append(v)
                              else:
                                  reports[c] = [v]
            clevel = nlevel
        """print("seller net:")
        print(seller_net)
        print("reports:")
        print(reports)
        print("bids:")
        print(bids)"""
        if len(seller_net)>0:
            """G1 = nx.DiGraph()

            # costruzione del grafo
            for bidder in seller_net:
                G1.add_edge('seller', bidder)

            for key in reports.keys():
                for value in reports[key]:
                    G1.add_edge(key, value)
            n_nodes = G1.number_of_nodes()-1"""
            allocation, payments = auction[2](self.k, seller_net, reports, bids)
            reward = sum(payments.values())
        else:
            #n_nodes = 0
            reward = 0
        #self.__n_nodes.append(n_nodes)
        """print("payments:")
        print(payments)
        print("allocation:")
        print(allocation)"""
        return reward
    

    def run(self, t, prob, val):
        t+=1
        a_t, auction_t = self.__init(t)
        #print("Selected arm at time ", t, " : ",a_t)
        reward = self.__receive_reward(a_t, auction_t, prob, val) #We save the reward assigned by the environment
        #print("recived reward: ", reward)
        # We update the number of times arm a_t has been chosen, its cumulative reward and its UCB value
        self.__num[a_t] += 1
        self.__rew[a_t] += reward
        self.__avgrew[a_t] = self.__rew[a_t]/self.__num[a_t]

        mu, sigma = self.__gaussian_dist[a_t]

        n = self.__num[a_t]
        #mu = (mu * (n-1) + reward) / n
        mu = self.__avgrew[a_t]
        sigma = (sigma * (n - 2) + (reward - mu) ** 2) / (n - 1) if n > 1 else 1
        self.__gaussian_dist[a_t] =  (mu, sigma)

        self.__ucb[a_t] = mu + (self.__ucb_scale * sigma)/ np.sqrt(n)

        return reward



class SocNetMec_UCB_mean:
    
    def __init__(self, G, T, k, auctions, arms_set, auction, gaussian_dist=None):
        self.G = G
        self.T = T
        self.k = k
        self.auctions = auctions
        self._type_auction = auction

        self.__arms_set = arms_set #initialize the set of arms
        self.__t = 0
        # Next initialization will serve for computing the best strategy during exploitation steps
        self.__num = {a:0 for a in self.__arms_set} #It saves the number of times arm a has been selected
        self.__rew = {a:0 for a in self.__arms_set} #It saves the cumulative reward achieved by arm a when selected
        self.__avgrew = {a:0 for a in self.__arms_set}  #It saves the average reward achieved by arm a until the current time step

        #It saves the ucb value of each arm until the current time step
        #It is initialised to infinity in order to allow that each arm is selected at least once
        if gaussian_dist is not None:
            self.__gaussian_dist = gaussian_dist
        else:
            self.__gaussian_dist = {a:(0,1) for a in arms_set}  

        self.__ucb = {a:float('inf') for a in self.__arms_set}
        #self.__n_nodes = []

    """def get_n(self):
        return self.__n_nodes"""
    def __init(self, t):
        self.__t = t
        a_t = max(self.__ucb, key=self.__ucb.get)  #We choose the arm that has the highest average revenue
        auction_t = self.auctions[self._type_auction]
        return a_t, auction_t

    def __invite(self, t, u, v, auction, prob, val):
        if prob(u,v):
          bv = val(t,v)
          sv = self.G[v]
          if not auction[0]:
            bv = random.uniform(bv*2/3, bv)
          if not auction[1]:
            new_sv = set()
            for n in sv:
                r = random.random()
                if r <= 0.15:
                    new_sv.add(n) 
            sv = new_sv
          return bv, sv
        else:
          return False, False
        
    def get_best_arm_approx(self):
        return self.__avgrew[max(self.__avgrew, key=self.__avgrew.get)]
        
    def __receive_reward(self, arm, auction, prob, val):
        """
        It select the reward for the given arm equal to the number of nodes in the graph
        that are reachable from the selected vertex x (the arm) only through alive edges

        Parameters
        ----------
        arm:
            the arm (vertex) that the learner want to use

        Returns
        ---------
            the reward assigned by the environment
        """
        clevel=[arm]
        visited=set(arm)
        invited=set(self.G[arm])
        seller_net = set()
        reports = dict()
        bids = dict()
        while len(clevel) > 0:
            nlevel=[]
            for c in clevel:
                for v in self.G[c]:
                    if v not in visited and v in invited:
                        bv,sv = self.__invite(self.__t-1,c,v,auction, prob, val)
                        #print("invited from ",v,": ",sv)
                        if bv:
                          visited.add(v)
                          nlevel.append(v)
                          for n in sv:
                            if n != arm:
                                invited.add(n)
                          bids[v] = bv
                          if c == arm:
                              seller_net.add(v)
                          else:
                              if c in reports:
                                  reports[c].append(v)
                              else:
                                  reports[c] = [v]
            clevel = nlevel
        """print("seller net:")
        print(seller_net)
        print("reports:")
        print(reports)
        print("bids:")
        print(bids)"""
        if len(seller_net)>0:
            """G1 = nx.DiGraph()

            # costruzione del grafo
            for bidder in seller_net:
                G1.add_edge('seller', bidder)

            for key in reports.keys():
                for value in reports[key]:
                    G1.add_edge(key, value)
            n_nodes = G1.number_of_nodes()-1"""
            allocation, payments = auction[2](self.k, seller_net, reports, bids)
            reward = sum(payments.values())
        else:
            #n_nodes = 0
            reward = 0
        #self.__n_nodes.append(n_nodes)
        """print("payments:")
        print(payments)
        print("allocation:")
        print(allocation)"""
        return reward
    

    def run(self, t, prob, val):
        t+=1
        a_t, auction_t = self.__init(t)
        ##print("Selected arm at time ", t, " : ",a_t)
        reward = self.__receive_reward(a_t, auction_t, prob, val) #We save the reward assigned by the environment
        ##print("recived reward: ", reward)
        # We update the number of times arm a_t has been chosen, its cumulative reward and its UCB value
        self.__num[a_t] += 1
        self.__rew[a_t] += reward
        self.__avgrew[a_t] = self.__rew[a_t]/self.__num[a_t]
        mu, sigma = self.__gaussian_dist[a_t]


        n = self.__num[a_t]
        mu = (mu * (n-1) + reward) / n
        sigma = (sigma * (n - 2) + (reward - mu) ** 2) / (n - 1) if n > 1 else 1
        self.__gaussian_dist[a_t] =  (mu, sigma)

        self.__ucb[a_t] = mu + math.sqrt(2*math.log(self.__t)/self.__num[a_t])


        return reward