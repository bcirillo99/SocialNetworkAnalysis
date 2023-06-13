import os
from matplotlib import pyplot as plt
import pandas as pd
for t in ["20000","40000","60000"]:
    fig, axs = plt.subplots(2, figsize=(10,6))
    for auction in ["MUDAN","GIDM"]:
        path = "results_armset_prior_2/"+t
        df = pd.read_csv(path+"/"+auction+"_2.csv")

        df_normal = df.loc[df['armset'] == "normal"].reset_index(drop=True)
        df_degree = df.loc[df['armset'] == "degree"].reset_index(drop=True)
        df_page = df.loc[df['armset'] == "pagerank"].reset_index(drop=True)
        df_vote = df.loc[df['armset'] == "voterank"].reset_index(drop=True)
        df_s_deg = df.loc[df['armset'] == "shapley_degree"].reset_index(drop=True)
        df_s_thresh = df.loc[df['armset'] == "shapley_threshold"].reset_index(drop=True)
        df_s_clos = df.loc[df['armset'] == "shapley_closeness"].reset_index(drop=True)


        results = []
        listk = [2,3,4,5]

        revenue_ucb_normal = []
        revenue_bayes_ucb_normal = []
        revenue_thompson_sampling_normal = []

        revenue_ucb_page = []
        revenue_bayes_ucb_page = []
        revenue_thompson_sampling_page = []

        revenue_ucb_degree = []
        revenue_bayes_ucb_degree = []
        revenue_thompson_sampling_degree = []

        revenue_ucb_vote = []
        revenue_bayes_ucb_vote = []
        revenue_thompson_sampling_vote = []

        revenue_ucb_s_deg = []
        revenue_bayes_ucb_s_deg = []
        revenue_thompson_sampling_s_deg = []


        revenue_ucb_s_thresh = []
        revenue_bayes_ucb_s_thresh = []
        revenue_thompson_sampling_s_thresh = []

        revenue_ucb_s_clos = []
        revenue_bayes_ucb_s_clos = []
        revenue_thompson_sampling_s_clos = []

        for k in listk:
            """df_normal_k = df_normal.loc[df_normal['k'] == k].reset_index(drop=True)
            revenue = df_normal_k['Revenue'].tolist()
            revenue_ucb_normal.append(revenue[0])"""

            df_degree_k = df_degree.loc[df_degree['k'] == k].reset_index(drop=True)
            revenue = df_degree_k['Revenue'].tolist()
            revenue_ucb_degree.append(revenue[0])

            df_page_k = df_page.loc[df_page['k'] == k].reset_index(drop=True)
            revenue = df_page_k['Revenue'].tolist()
            revenue_ucb_page.append(revenue[0])

            df_vote_k = df_vote.loc[df_vote['k'] == k].reset_index(drop=True)
            revenue = df_vote_k['Revenue'].tolist()
            revenue_ucb_vote.append(revenue[0])

            df_s_deg_k = df_s_deg.loc[df_s_deg['k'] == k].reset_index(drop=True)
            revenue = df_s_deg_k['Revenue'].tolist()
            revenue_ucb_s_deg.append(revenue[0])
            """
            df_s_thresh_k = df_s_thresh.loc[df_s_thresh['k'] == k].reset_index(drop=True)
            revenue = df_s_thresh_k['Revenue'].tolist()
            revenue_ucb_s_thresh.append(revenue[0])

            df_s_clos_k = df_s_clos.loc[df_s_clos['k'] == k].reset_index(drop=True)
            revenue = df_s_clos_k['Revenue'].tolist()
            revenue_ucb_s_clos.append(revenue[0])"""

        


        

        if auction =="MUDAN":
            print(revenue_bayes_ucb_degree)
            axs[0].plot(listk, revenue_ucb_degree, label = 'Degree')
            axs[0].plot(listk, revenue_ucb_page, label = 'PageRank')
            axs[0].plot(listk, revenue_ucb_vote, label = 'VoteRank')
            axs[0].plot(listk, revenue_ucb_s_deg, label = 's. degree')
            axs[0].set_xlabel('k')
            axs[0].set_xticks(listk, listk)
            axs[0].set_ylabel('Revenue')
            axs[0].set_title(auction+ " (UCB)")
            axs[0].legend()
            

        else:
            print(revenue_ucb_degree)
            axs[1].plot(listk, revenue_ucb_degree, label = 'Degree')
            axs[1].plot(listk, revenue_ucb_page, label = 'PageRank')
            axs[1].plot(listk, revenue_ucb_vote, label = 'VoteRank')
            axs[1].plot(listk, revenue_ucb_s_deg, label = 's. degree')
            axs[1].set_xlabel('k')
            axs[1].set_xticks(listk, listk)
            axs[1].set_ylabel('Revenue')
            axs[1].set_title(auction+ " (UCB)")
            axs[1].legend()

    fig.tight_layout(pad=0.5)
    if not os.path.exists(path+'/plot/'):
        os.makedirs(path+'/plot/')
    plt.savefig(path+'/plot/bandit_charts_'+t+'_2.png')

    print(listk)


    print(df_page_k.reset_index(drop=True))