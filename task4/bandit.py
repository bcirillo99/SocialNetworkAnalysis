import copy
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import math
import networkx as nx
import random
# This is a sample Python script.
import sys
# adding Folder_2 to the system path
sys.path.insert(0, '../')
from utils import *
from operator import itemgetter

    
class Social_Environment:
    """
    An environment that returns as the reward of arm x (the selected vertex) the number of nodes in the social network
    that are reachable from the selected vertex x only through alive edges
    """
    def __init__(self, G, prob_dict):
        """
        We initialise the mean of the Bernoulli distribution for each arm

        Parameters
        ----------
        G: nx.Graph or nx.DiGraph
            A networkx undirected or directed graphs

        means_dict: dict
            dictionary whose keys are edges (u,v) and whose values the respective probability to be alive p(u,v)
        """
        self.__G = G
        self.__prob_dict = prob_dict
    
    def __is_alive(self, u,v):
        """
        The environment set each edge (u,v) as alive with probability p(u,v), and dead otherwise
        """
        e = (u,v)
        if not self.__G.is_directed():
            if e not in self.__prob_dict:
                e = (v,u)
        r = random.random()
        if r <= self.__prob_dict[e]: #with probability p(u,v)
            return True
        else:
            return False


    def receive_reward(self, arm):
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
        while len(clevel) > 0:
            nlevel=[]
            for c in clevel:
                for v in self.__G[c]:
                    if v not in visited and self.__is_alive(c,v):
                        visited.add(v)
                        nlevel.append(v)
            clevel = nlevel
        return len(visited)-1

    
class EpsGreedy_Learner:
    def __init__(self, arms_set, environment, eps):
        self.__arms_set = arms_set #initialize the set of arms
        self.__environment = environment #initialize the environment
        self.__eps = eps #initialize the sequence of eps_t

        #Next initialization will serve for computing the best strategy during exploitation steps
        self.__num = {a:0 for a in arms_set} #It saves the number of times arm a has been selected
        self.__rew = {a:0 for a in arms_set} #It saves the cumulative reward achieved by arm a when selected
        self.__avgrew = {a:0 for a in arms_set}  #It saves the average reward achieved by arm a until the current time step
        self.__t = 0 #It saves the current time step

    #This function returns the arm chosen by the learner and the corresponding reward returned by the environment
    def play_arm(self):
        r = random.random()
        if r <= self.__eps(self.__t):
            a_t = random.choice(self.__arms_set) #We choose an arm uniformly at random
        else:
            a_t = max(self.__avgrew, key=self.__avgrew.get) #We choose the arm that has the highest average revenue
        reward = self.__environment.receive_reward(a_t) #We save the reward assigned by the environment
        
        #We update the number of times arm a_t has been chosen, its cumulative and its average reward
        self.__num[a_t] += 1
        self.__rew[a_t] += reward
        self.__avgrew[a_t] = self.__rew[a_t] / self.__num[a_t]
        self.__t += 1 #We are ready for a new time step

        return a_t, reward
    
    def get_best_arm_approx(self):
        return self.__avgrew[max(self.__avgrew, key=self.__avgrew.get)]

class UCB_Learner:

    def __init__(self, arms_set, environment): 
        self.__arms_set = arms_set #initialize the set of arms
        self.__environment = environment #initialize the environment
        self.__t = 1 #UNCOMMENT THIS FOR UNKNOWN TIME HORIZON
        # Next initialization will serve for computing the best strategy during exploitation steps
        self.__num = {a:0 for a in arms_set} #It saves the number of times arm a has been selected
        self.__rew = {a:0 for a in arms_set} #It saves the cumulative reward achieved by arm a when selected
        self.__avgrew = {a:0 for a in arms_set}  #It saves the average reward achieved by arm a until the current time step

        #It saves the ucb value of each arm until the current time step
        #It is initialised to infinity in order to allow that each arm is selected at least once
        self.__ucb = {a:float('inf') for a in arms_set}

    # This function returns the arm chosen by the learner and the corresponding reward returned by the environment
    def play_arm(self):
        a_t = max(self.__ucb, key=self.__ucb.get)  #We choose the arm that has the highest average revenue
        reward = self.__environment.receive_reward(a_t) #We save the reward assigned by the environment
        # We update the number of times arm a_t has been chosen, its cumulative reward and its UCB value
        self.__num[a_t] += 1
        self.__rew[a_t] += reward
        self.__avgrew[a_t] = self.__rew[a_t]/self.__num[a_t]
        print("T: ", self.__t)
        self.__ucb[a_t] = self.__avgrew[a_t] + math.sqrt(2*math.log(self.__t)/self.__num[a_t])
        self.__t += 1

        return a_t, reward
    
    def get_best_arm_approx(self):
        return self.__avgrew[max(self.__avgrew, key=self.__avgrew.get)]

class Thompson_Learner:
    def __init__(self, arms_set, environment): 
        self.__arms_set = arms_set #initialize the set of arms
        self.__environment = environment #initialize the environment
        self.__t = 1 #UNCOMMENT THIS FOR UNKNOWN TIME HORIZON
        # Next initialization will serve for computing the best strategy during exploitation steps
        self.__num = {a:0 for a in arms_set} #It saves the number of times arm a has been selected
        self.__rew = {a:0 for a in arms_set} #It saves the cumulative reward achieved by arm a when selected
        self.__avgrew = {a:0 for a in arms_set}  #It saves the average reward achieved by arm a until the current time step

        #It saves the alfa and beta values of the beta distribution of each arm until the current time step
        #It is initialised to (1,1) in order to have a uniform prior for each arm
        self.__beta_dist = {a:(1,1) for a in arms_set}  


    # This function returns the arm chosen by the learner and the corresponding reward returned by the environment
    def play_arm(self):
        beta_max = 0
        for arm in self.__arms_set:
            alfa, beta = self.__beta_dist[arm]
            beta_value = random.betavariate(alfa, beta)
            if beta_value > beta_max:
                beta_max = beta_value
                a_t = arm
        reward = self.__environment.receive_reward(a_t) #We save the reward assigned by the environment
        # We update the number of times arm a_t has been chosen, its cumulative reward and its UCB value
        self.__num[a_t] += 1
        self.__rew[a_t] += reward
        self.__avgrew[a_t] = self.__rew[a_t]/self.__num[a_t]
        alfa, beta = self.__beta_dist[arm]
        alfa += reward
        # beta += max(1 - reward, 0)
        beta += 1
        self.__t += 1
        # manca aggiornamento beta

        return a_t, reward
    
    def get_best_arm_approx(self):
        return self.__avgrew[max(self.__avgrew, key=self.__avgrew.get)]
    
class Gaussian_Thompson_Learner:
    def __init__(self, arms_set, environment, gaussian_dist=None): 
        self.__arms_set = arms_set #initialize the set of arms
        self.__environment = environment #initialize the environment
        self.__t = 1 #UNCOMMENT THIS FOR UNKNOWN TIME HORIZON
        # Next initialization will serve for computing the best strategy during exploitation steps
        self.__num = {a:0 for a in arms_set} #It saves the number of times arm a has been selected
        self.__rew = {a:0 for a in arms_set} #It saves the cumulative reward achieved by arm a when selected
        self.__avgrew = {a:0 for a in arms_set}  #It saves the average reward achieved by arm a until the current time step

        #It saves the mean and variance values for each arm until in order to have a prior gaussian standard
        if gaussian_dist is not None:
            self.__gaussian_dist = gaussian_dist
        else:
            self.__gaussian_dist = {a:(0,1) for a in arms_set}  


    # This function returns the arm chosen by the learner and the corresponding reward returned by the environment
    def play_arm(self):
        max = 0
        for arm in self.__arms_set:
            mu, sigma = self.__gaussian_dist[arm]
            gauss_value = np.random.normal(mu, sigma)
            if gauss_value > max:
                max = gauss_value
                a_t = arm
        reward = self.__environment.receive_reward(a_t) #We save the reward assigned by the environment
        # We update the number of times arm a_t has been chosen, its cumulative reward and its UCB value
        self.__num[a_t] += 1
        self.__rew[a_t] += reward
        self.__avgrew[a_t] = self.__rew[a_t]/self.__num[a_t]
        mu, sigma = self.__gaussian_dist[arm]

        n = self.__num[a_t]
        mu = (mu * (n-1) + reward) / n
        sigma = (sigma * (n - 2) + (reward - mu) ** 2) / (n - 1) if n > 1 else 1

        self.__gaussian_dist[a_t] =  (mu, sigma)
        self.__t += 1

        return a_t, reward
    
    
    def get_best_arm_approx(self):
        return self.__avgrew[max(self.__avgrew, key=self.__avgrew.get)]
    


if __name__ == '__main__':
    #INITIALIZATION

    parser = ArgumentParser()
    parser.add_argument('--directed', help='type of Graph', type=bool, default=False)
    parser.add_argument('--T', help='Time Horizon', type=int, default=10)
    parser.add_argument('--N', help='number of simulations', type=int, default=1)
    parser.add_argument('--file_name', help='noem file di testo su cui salvare i risultati', type=str, default="centrality.txt")
    
    args = parser.parse_args()

    directed = args.directed
    
    #We would like to evaluate the expected regret with respect to t
    #To this aim, we cannot just run a single simulation:
    #the result can be biased by extreme random choices (of both the environment and the learner)
    #For this reason we run N simulations,
    #and we will evaluate against t the average regret over the N simulations
    #To this aim, we define N, and we will record for each learner a matrix containing
    #the regret for each simulation and each time step within the simulation

    #We need the time horizon to have a set number of attempts anyway, we can't make it go to infinity
    T = args.T #Time Horizon
    N = args.N #number of simulations

    file_name = args.file_name
    
    if directed:
        #G = create_graph_from_txt('../data/Cit-HepTh.txt', sep='\t', directed=True)
        G = create_graph_from_csv('../data/soc-sign-bitcoinalpha.csv', directed=True)
        G = create_graph_from_txt('../data/retweet.txt', sep=' ', directed=True)
        """G = nx.DiGraph()
        G.add_edge('A', 'B')
        G.add_edge('A','D')
        G.add_edge('B', 'E')
        G.add_edge('B', 'C')
        G.add_edge('C', 'E')
        G.add_edge('C', 'D')
        G.add_edge('C', 'F')"""
    else:
        #G = create_graph_from_csv('../data/food_facebook.csv')
        G = create_graph_from_txt('../data/dolphins.txt', sep=' ')
        """G = nx.Graph()
        G.add_edge('A', 'B')
        G.add_edge('A', 'C')
        G.add_edge('A', 'D')
        G.add_edge('B', 'E')
        G.add_edge('D', 'E')
        G.add_edge('E', 'F')
        G.add_edge('E', 'G')
        G.add_edge('D', 'H')
        G.add_edge('H', 'I')
        G.add_edge('I', 'L')
        G.add_edge('L', 'D')
        G.add_edge('L', 'A')
        G.add_edge('A', 'M')
        G.add_edge('M', 'N')
        G.add_edge('M', 'Q')
        G.add_edge('Q', 'P')
        G.add_edge('P', 'N')
        G.add_edge('P', 'O')
        G.add_edge('O', 'N')"""

    s = f'\ Network ---> directed: {G.is_directed()}, node: {G.number_of_nodes()}, edges: {G.number_of_edges()}'
    print(s)
    #Set of Arms
    arms_set = [v for v in G.nodes()]
    #Means the Bernoulli Stochastic Environment
    prob_dict = {e:random.random() for e in G.edges()}

    """
    print(arms_set)
    print(prob_dict)
    """

    
    eps_regrets = {n: {t: 0 for t in range(T)} for n in range(N)} #regret matrix for the eps-greedy learner
    ucb_regrets = {n: {t: 0 for t in range(T)} for n in range(N)} #regret matrix for the UCB learner
    thompson_regrets = {n: {t: 0 for t in range(T)} for n in range(N)} #regret matrix for the thompson learner


    #INITIALIZATION FOR EPS-GREEDY
    """
    A common choice for eps_t = (K log t/t)^1/3

    # FOR UNKNOWN TIME HORIZON COMMENT THE FOLLOWING LINES
    eps = [1] #for the first step we cannot make exploitation, so eps_1 = 1
    eps.extend((len(arms_set)*math.log(t)/t)**(1/3) for t in range(2,T+1))
    #FOR UNKNOWN TIME HORIZON UNCOMMENT THE FOLLOWING LINES
    """
    def give_eps(t):
       if t == 0:
           return 1  #for the first step we cannot make exploitation, so eps_1 = 1
       return (len(arms_set)*math.log(t+1)/(t+1))**(1/3)
    
    eps = give_eps

    #SIMULATION PLAY
    for n in tqdm(range(N)):
        ucb_cum_reward = 0 #it saves the cumulative reward of the UCB learner
        eps_cum_reward = 0 #it saves the cumulative reward of the eps-greedy learner
        thompson_cum_reward = 0 #it saves the cumulative reward of the thompson learner 

        cum_ucb_opt_reward = 0 #it saves the cumulative reward of the UCB learner best-arm in hindsight
        cum_eps_opt_reward = 0 #it saves the cumulative reward of the eps-greedy learner best-arm in hindsight
        cum_thompson_opt_reward = 0 #it saves the cumulative reward of the eps-greedy learner best-arm in hindsight

        dict_count_best_arm_eps = {a:0 for a in arms_set}
        dict_count_best_arm_ucb = {a:0 for a in arms_set}
        dict_count_best_arm_thompson = {a:0 for a in arms_set}
        #Environment
        env = Social_Environment(G,prob_dict)
        #Eps-Greedy Learner
        eps_learn = EpsGreedy_Learner(arms_set, env, eps) #COMMENT FOR UNKNOWN TIME HORIZON
        #eps_learn = EpsGreedy_Learner(arms_set, env, eps) #UNCOMMENT FOR UNKNOWN TIME HORIZON
        #UCB Learner
        ucb_learn = UCB_Learner(arms_set, env)  #COMMENT FOR UNKNOWN TIME HORIZON
        #ucb_learn = UCB_Learner(arms_set, env)  #UNCOMMENT FOR UNKNOWN TIME HORIZON
        thompson_learn = Thompson_Learner(arms_set, env)  #COMMENT FOR UNKNOWN TIME HORIZON

        for t in tqdm(range(T)):
            
            #reward obtained by the eps_greedy learner
            print(t)
            a, reward = eps_learn.play_arm()
            eps_cum_reward += reward
            a_eps = a
            dict_count_best_arm_eps[a] += 1

            # reward obtained by the ucb learner
            a, reward = ucb_learn.play_arm()
            ucb_cum_reward += reward
            a_ucb = a
            dict_count_best_arm_ucb[a] += 1

            # reward obtained by the thompson learner
            a, reward = thompson_learn.play_arm()
            thompson_cum_reward += reward
            a_thompson = a
            dict_count_best_arm_thompson[a] += 1

            cum_eps_opt_reward += eps_learn.get_best_arm_approx()
            cum_ucb_opt_reward += ucb_learn.get_best_arm_approx()
            cum_thompson_opt_reward += thompson_learn.get_best_arm_approx()
            """opt_a_eps = eps_learn.get_best_arm_approx()
            opt_a_ucb = ucb_learn.get_best_arm_approx()
            opt_a_thompson = thompson_learn.get_best_arm_approx()


            #reward obtained by the optimal arm for eps-greedy
            cum_eps_opt_reward += env.receive_reward(opt_a_eps)

            #reward obtained by the optimal arm for ucb
            cum_ucb_opt_reward += env.receive_reward(opt_a_ucb)

            #reward obtained by the optimal arm for thompson
            cum_thompson_opt_reward += env.receive_reward(opt_a_thompson)"""

            #regret of the eps_greedy learner
            eps_regrets[n][t] = cum_eps_opt_reward - eps_cum_reward
            #regret of the ucb learner
            ucb_regrets[n][t] = cum_ucb_opt_reward - ucb_cum_reward
            #regret of the thompson learner
            thompson_regrets[n][t] = cum_thompson_opt_reward - thompson_cum_reward
        
        print("PROVA ", n," :")
        print("eps-greedy: ",a_eps,env.receive_reward(a_eps))
        print(dict(sorted(dict_count_best_arm_eps.items(), key=itemgetter(1), reverse=True)[:5]))
        print("UCB1: ",a_ucb,env.receive_reward(a_ucb))
        print(dict(sorted(dict_count_best_arm_ucb.items(), key=itemgetter(1), reverse=True)[:5]))
        print("thompson: ",a_thompson,env.receive_reward(a_thompson))
        print(dict(sorted(dict_count_best_arm_thompson.items(), key=itemgetter(1), reverse=True)[:5]))
        print("\n\n----------------------\n\n")
    
    #compute the mean regret of the eps greedy and ucb learner and thompson
    eps_mean_regrets = {t:0 for t in range(T)}
    ucb_mean_regrets = {t:0 for t in range(T)}
    thompson_mean_regrets = {t:0 for t in range(T)}
    for t in range(T):
        eps_mean_regrets[t] = sum(eps_regrets[n][t] for n in range(N))/N
        ucb_mean_regrets[t] = sum(ucb_regrets[n][t] for n in range(N))/N
        thompson_mean_regrets[t] = sum(thompson_regrets[n][t] for n in range(N))/N

    #VISUALIZATION OF RESULTS
    #compute t^2/3 (c K log t)^1/3
    ref_eps = list()
    for t in range(1, T+1):
        ref_eps.append((t**(2/3))*(2*len(arms_set)*math.log(t))**(1/3))

    #compute c*sqrt(KtlogT)
    ref_ucb = list()
    for t in range(1, T+1):
        ref_ucb.append(math.sqrt(len(arms_set)*t*math.log(T)))
    
    #compute c*sqrt(KtlogT)
    ref_thompson = list()
    for t in range(1, T+1):
        ref_thompson.append(math.sqrt(len(arms_set)*t*math.log(T)))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    fig.suptitle(s, fontsize=10)
    #Plot eps-greedy regret against its reference value
    ax1.plot(range(1,T+1), eps_mean_regrets.values(), label = 'eps_mean_regret')
    ax1.plot(range(1,T+1), ref_eps, label = f't^2/3 (2 K log t)^1/3')
    ax1.set_xlabel('t')
    ax1.set_ylabel('E[R(t)]')
    ax1.legend()

    #Plot ucb regret against its reference value
    ax2.plot(range(1,T+1), ucb_mean_regrets.values(), label = 'ucb_mean_regret')
    ax2.plot(range(1,T+1), ref_ucb, label = f'sqrt(K*t*logT)')
    ax2.set_xlabel('t')
    ax2.set_ylabel('E[R(t)]')
    ax2.legend()

    #Plot thompson regret against its reference value
    ax3.plot(range(1,T+1), thompson_mean_regrets.values(), label = 'thompson_mean_regret')
    ax3.plot(range(1,T+1), ref_thompson, label = f'sqrt(K*T*logT)')
    ax3.set_xlabel('t')
    ax3.set_ylabel('E[R(t)]')
    ax3.legend()

    #Plot ucb regret against eps-greedy and thompson regret
    ax4.plot(range(1,T+1), eps_mean_regrets.values(), label = 'eps_mean_regret')
    ax4.plot(range(1,T+1), ucb_mean_regrets.values(), label = 'ucb_mean_regret')
    ax4.plot(range(1,T+1), thompson_mean_regrets.values(), label = 'thompson_mean_regret')
    ax4.set_xlabel('t')
    ax4.set_ylabel('E[R(t)]')
    ax4.legend()

    plt.show()
