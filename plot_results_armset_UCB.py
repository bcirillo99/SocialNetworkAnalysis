import os
from matplotlib import pyplot as plt
import pandas as pd



results = []
listk = [1,2,3,4,5]


revenue_ucb_page = []
revenue_ucb_40000_page = []
revenue_ucb_60000_page = []
revenue_ucb_80000_page = []

revenue_ucb_degree = []
revenue_ucb_40000_degree = []
revenue_ucb_60000_degree = []
revenue_ucb_80000_degree = []

revenue_ucb_vote = []
revenue_ucb_40000_vote = []
revenue_ucb_60000_vote = []
revenue_ucb_80000_vote = []

for n in ["20000","40000","60000","80000"]:

    path = "final_results_UCB/"+n+"/"
    df = pd.read_csv(path+"MUDAR.csv")


    df_degree = df.loc[df['armset'] == "degree"].reset_index(drop=True)
    df_page = df.loc[df['armset'] == "pagerank"].reset_index(drop=True)
    #df_vcg = pd.read_csv("final_results/80000/VCG.csv")
    df_vote = df.loc[df['armset'] == "voterank"].reset_index(drop=True)

    """df_degree = df.loc[df['Auction'] == "Degree"].reset_index(drop=True)
    df_page = df.loc[df['Auction'] == "PageRank"].reset_index(drop=True)
    df_mixed = df.loc[df['Auction'] == "MIXED"].reset_index(drop=True)"""



    for k in listk:
        
        df_degree_k = df_degree.loc[df_degree['k'] == k].reset_index(drop=True)
        revenue = df_degree_k['Revenue'].tolist()
        if n == "20000":
            revenue_ucb_degree.append(revenue[0])
        elif n == "40000":
            revenue_ucb_40000_degree.append(revenue[0])
        elif n == "60000":
            revenue_ucb_60000_degree.append(revenue[0])
        else:
            revenue_ucb_80000_degree.append(revenue[0])

        df_page_k = df_page.loc[df_page['k'] == k].reset_index(drop=True)
        revenue = df_page_k['Revenue'].tolist()
        if n == "20000":
            revenue_ucb_page.append(revenue[0])
        elif n == "40000":
            revenue_ucb_40000_page.append(revenue[0])
        elif n == "60000":
            revenue_ucb_60000_page.append(revenue[0])
        else:
            revenue_ucb_80000_page.append(revenue[0])

        """df_vcg_k = df_vcg.loc[df_vcg['k'] == k].reset_index(drop=True)
        revenue = df_vcg_k['Revenue'].tolist()
        revenue_ucb_vcg.append(revenue[0])
        revenue_ucb_40000_vcg.append(revenue[2])
        revenue_eps_vcg.append(revenue[3])
        revenue_thompson_vcg.append(revenue[4])"""

        df_vote_k = df_vote.loc[df_vote['k'] == k].reset_index(drop=True)
        revenue = df_vote_k['Revenue'].tolist()
        if n == "20000":
            revenue_ucb_vote.append(revenue[0])
        elif n == "40000":
            revenue_ucb_40000_vote.append(revenue[0])
        elif n == "60000":
            revenue_ucb_60000_vote.append(revenue[0])
        else:
            revenue_ucb_80000_vote.append(revenue[0])
    
    

fig, axs = plt.subplots(4, figsize=(10,6))

axs[0].plot(listk, revenue_ucb_degree, label = 'Degree')
axs[0].plot(listk, revenue_ucb_page, label = 'PageRank')
axs[0].plot(listk, revenue_ucb_vote, label = 'VoteRank')
axs[0].set_xlabel('k')
axs[0].set_xticks(listk, listk)
axs[0].set_ylabel('Revenue')
axs[0].set_title("UCB T=20000")
axs[0].legend()

axs[1].plot(listk, revenue_ucb_40000_degree, label = 'Degree')
axs[1].plot(listk, revenue_ucb_40000_page, label = 'PageRank')
axs[1].plot(listk, revenue_ucb_40000_vote, label = 'VoteRank')
axs[1].set_xlabel('k')
axs[1].set_xticks(listk, listk)
axs[1].set_ylabel('Revenue')
axs[1].set_title("UCB T=40000")
axs[1].legend()

axs[2].plot(listk, revenue_ucb_60000_degree, label = 'Degree')
axs[2].plot(listk, revenue_ucb_60000_page, label = 'PageRank')
axs[2].plot(listk, revenue_ucb_60000_vote, label = 'VoteRank')
axs[2].set_xlabel('k')
axs[2].set_xticks(listk, listk)
axs[2].set_ylabel('Revenue')
axs[2].set_title("UCB T=60000")
axs[2].legend()

axs[3].plot(listk, revenue_ucb_80000_degree, label = 'Degree')
axs[3].plot(listk, revenue_ucb_80000_page, label = 'PageRank')
axs[3].plot(listk, revenue_ucb_80000_vote, label = 'VoteRank')
axs[3].set_xlabel('k')
axs[3].set_xticks(listk, listk)
axs[3].set_ylabel('Revenue')
axs[3].set_title("UCB T=80000")
axs[3].legend()

fig.tight_layout(pad=0.5)
path = "final_results_UCB/"
plt.savefig(path+'armset_charts.png')


