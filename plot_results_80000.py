import os
from matplotlib import pyplot as plt
import pandas as pd



df_mudan = pd.read_csv("final_results/80000/MUDAN.csv")
df_mudar = pd.read_csv("final_results/80000/MUDAR.csv")
#df_vcg = pd.read_csv("final_results/80000/VCG.csv")
df_gidm = pd.read_csv("final_results/80000/GIDM.csv")

"""df_mudan = df.loc[df['Auction'] == "MUDAN"].reset_index(drop=True)
df_mudar = df.loc[df['Auction'] == "MUDAR"].reset_index(drop=True)
df_mixed = df.loc[df['Auction'] == "MIXED"].reset_index(drop=True)"""

results = []
listk = [1,2,3,4,5]

path = "final_results/80000/"

revenue_ucb_mudar = []
revenue_bayes_ucb_mudar = []
revenue_eps_mudar = []
revenue_thompson_mudar = []

revenue_ucb_mudan = []
revenue_bayes_ucb_mudan = []
revenue_eps_mudan = []
revenue_thompson_mudan = []

revenue_ucb_vcg = []
revenue_bayes_ucb_vcg = []
revenue_eps_vcg = []
revenue_thompson_vcg = []

revenue_ucb_gidm = []
revenue_bayes_ucb_gidm = []
revenue_eps_gidm = []
revenue_thompson_gidm = []
for k in listk:
    
    df_mudan_k = df_mudan.loc[df_mudan['k'] == k].reset_index(drop=True)
    revenue = df_mudan_k['Revenue'].tolist()
    revenue_ucb_mudan.append(revenue[0])
    revenue_bayes_ucb_mudan.append(revenue[1])
    """revenue_eps_mudan.append(revenue[2])
    revenue_thompson_mudan.append(revenue[3])"""
    revenue_thompson_mudan.append(revenue[2])

    df_mudar_k = df_mudar.loc[df_mudar['k'] == k].reset_index(drop=True)
    revenue = df_mudar_k['Revenue'].tolist()
    revenue_ucb_mudar.append(revenue[0])
    revenue_bayes_ucb_mudar.append(revenue[1])
    #revenue_eps_mudar.append(revenue[2])
    revenue_thompson_mudar.append(revenue[2])

    """df_vcg_k = df_vcg.loc[df_vcg['k'] == k].reset_index(drop=True)
    revenue = df_vcg_k['Revenue'].tolist()
    revenue_ucb_vcg.append(revenue[0])
    revenue_bayes_ucb_vcg.append(revenue[2])
    revenue_eps_vcg.append(revenue[3])
    revenue_thompson_vcg.append(revenue[4])"""

    df_gidm_k = df_gidm.loc[df_gidm['k'] == k].reset_index(drop=True)
    revenue = df_gidm_k['Revenue'].tolist()
    revenue_ucb_gidm.append(revenue[0])
    revenue_bayes_ucb_gidm.append(revenue[1])
    #revenue_eps_gidm.append(revenue[2])
    revenue_thompson_gidm.append(revenue[2])


fig, axs = plt.subplots(3, figsize=(10,6))

axs[0].plot(listk, revenue_ucb_mudan, label = 'UCB')
axs[0].plot(listk, revenue_bayes_ucb_mudan, label = 'Bayesian UCB')
#axs[0].plot(listk, revenue_eps_mudan, label = 'Epsilon Greedy')
axs[0].plot(listk, revenue_thompson_mudan, label = 'Thompson Sampling')
axs[0].set_xlabel('k')
axs[0].set_xticks(listk, listk)
axs[0].set_ylabel('Revenue')
axs[0].set_title("MUDAN")
axs[0].legend()

axs[1].plot(listk, revenue_ucb_mudar, label = 'UCB')
axs[1].plot(listk, revenue_bayes_ucb_mudar, label = 'Bayesian UCB')
#axs[1].plot(listk, revenue_eps_mudar, label = 'Epsilon Greedy')
axs[1].plot(listk, revenue_thompson_mudar, label = 'Thompson Sampling')
axs[1].set_xlabel('k')
axs[1].set_xticks(listk, listk)
axs[1].set_ylabel('Revenue')
axs[1].set_title("MUDAR")
axs[1].legend()


axs[2].plot(listk, revenue_ucb_gidm, label = 'UCB')
axs[2].plot(listk, revenue_bayes_ucb_gidm, label = 'Bayesian UCB')
#axs[2].plot(listk, revenue_eps_gidm, label = 'Epsilon Greedy')
axs[2].plot(listk, revenue_thompson_gidm, label = 'Thompson Sampling')
axs[2].set_xlabel('k')
axs[2].set_xticks(listk, listk)
axs[2].set_ylabel('Revenue')
axs[2].set_title("GIDM")
axs[2].legend()

fig.tight_layout(pad=0.5)
plt.savefig(path+'auctions_chart.png')
#plt.show()

fig, axs = plt.subplots(3, figsize=(10,6))

axs[0].plot(listk, revenue_ucb_mudan, label = 'MUDAN')
axs[0].plot(listk, revenue_ucb_mudar, label = 'MUDAR')
axs[0].plot(listk, revenue_ucb_gidm, label = 'GIDM')
axs[0].set_xlabel('k')
axs[0].set_xticks(listk, listk)
axs[0].set_ylabel('Revenue')
axs[0].set_title("UCB")
axs[0].legend()

axs[1].plot(listk, revenue_bayes_ucb_mudan, label = 'MUDAN')
axs[1].plot(listk, revenue_bayes_ucb_mudar, label = 'MUDAR')
axs[1].plot(listk, revenue_bayes_ucb_gidm, label = 'GIDM')
axs[1].set_xlabel('k')
axs[1].set_xticks(listk, listk)
axs[1].set_ylabel('Revenue')
axs[1].set_title("Bayesian UCB")
axs[1].legend()

"""axs[1,0].plot(listk, revenue_eps_mudan, label = 'MUDAN')
axs[1,0].plot(listk, revenue_eps_mudar, label = 'MUDAR')
axs[1,0].plot(listk, revenue_eps_gidm, label = 'GIDM')
axs[1,0].set_xlabel('k')
axs[1,0].set_xticks(listk, listk)
axs[1,0].set_ylabel('Revenue')
axs[1,0].set_title("EPS-Greedy")
axs[1,0].legend()"""

axs[2].plot(listk, revenue_thompson_mudan, label = 'MUDAN')
axs[2].plot(listk, revenue_thompson_mudar, label = 'MUDAR')
axs[2].plot(listk, revenue_thompson_gidm, label = 'GIDM')
axs[2].set_xlabel('k')
axs[2].set_xticks(listk, listk)
axs[2].set_ylabel('Revenue')
axs[2].set_title("Thompson Sampling")
axs[2].legend()

fig.tight_layout(pad=0.5)
plt.savefig(path+'bandit_charts.png')

print(listk)
#plt.show()


print(df_mudar_k.reset_index(drop=True))