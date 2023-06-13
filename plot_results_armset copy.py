import os
from matplotlib import pyplot as plt
import pandas as pd
for t in ["20000","60000"]:
    fig, axs = plt.subplots(2, figsize=(10,6))
    for auction in ["MUDAN","GIDM"]:
        path = "results_armset_prior/"+t
        df = pd.read_csv(path+"/"+auction+"_2.csv")

        df_normal = df.loc[df['armset'] == "normal"].reset_index(drop=True)
        df_degree = df.loc[df['armset'] == "degree"].reset_index(drop=True)
        df_page = df.loc[df['armset'] == "pagerank"].reset_index(drop=True)
        #df_vcg = pd.read_csv("final_results/60000/VCG.csv")
        df_vote = df.loc[df['armset'] == "voterank"].reset_index(drop=True)
        df_s_deg = df.loc[df['armset'] == "shapley_degree"].reset_index(drop=True)
        df_s_thresh = df.loc[df['armset'] == "shapley_threshold"].reset_index(drop=True)
        df_s_clos = df.loc[df['armset'] == "shapley_closeness"].reset_index(drop=True)

        """df_degree = df.loc[df['Auction'] == "Degree"].reset_index(drop=True)
        df_page = df.loc[df['Auction'] == "PageRank"].reset_index(drop=True)
        df_mixed = df.loc[df['Auction'] == "MIXED"].reset_index(drop=True)"""

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
            df_normal_k = df_normal.loc[df_normal['k'] == k].reset_index(drop=True)
            revenue = df_normal_k['Revenue'].tolist()
            revenue_ucb_normal.append(revenue[0])
            """revenue_bayes_ucb_normal.append(revenue[1])
            revenue_thompson_sampling_normal.append(revenue[2])"""

            df_degree_k = df_degree.loc[df_degree['k'] == k].reset_index(drop=True)
            revenue = df_degree_k['Revenue'].tolist()
            revenue_ucb_degree.append(revenue[0])
            """revenue_bayes_ucb_degree.append(revenue[1])
            revenue_thompson_sampling_degree.append(revenue[3])"""

            df_page_k = df_page.loc[df_page['k'] == k].reset_index(drop=True)
            revenue = df_page_k['Revenue'].tolist()
            revenue_ucb_page.append(revenue[0])
            """revenue_bayes_ucb_page.append(revenue[1])
            revenue_thompson_sampling_page.append(revenue[3])"""

            """df_vcg_k = df_vcg.loc[df_vcg['k'] == k].reset_index(drop=True)
            revenue = df_vcg_k['Revenue'].tolist()
            revenue_ucb_vcg.append(revenue[0])
            revenue_bayes_ucb_vcg.append(revenue[3])
            revenue_eps_vcg.append(revenue[3])
            revenue_thompson_vcg.append(revenue[4])"""

            df_vote_k = df_vote.loc[df_vote['k'] == k].reset_index(drop=True)
            revenue = df_vote_k['Revenue'].tolist()
            revenue_ucb_vote.append(revenue[0])
            """revenue_bayes_ucb_vote.append(revenue[1])
            revenue_thompson_sampling_vote.append(revenue[3])"""

            df_s_deg_k = df_s_deg.loc[df_s_deg['k'] == k].reset_index(drop=True)
            revenue = df_s_deg_k['Revenue'].tolist()
            revenue_ucb_s_deg.append(revenue[0])

            df_s_thresh_k = df_s_thresh.loc[df_s_thresh['k'] == k].reset_index(drop=True)
            revenue = df_s_thresh_k['Revenue'].tolist()
            revenue_ucb_s_thresh.append(revenue[0])

            df_s_clos_k = df_s_clos.loc[df_s_clos['k'] == k].reset_index(drop=True)
            revenue = df_s_clos_k['Revenue'].tolist()
            revenue_ucb_s_clos.append(revenue[0])
            

        """
        fig, axs = plt.subplots(6, figsize=(10,10))

        axs[0].plot(listk, revenue_ucb_degree, label = 'UCB')
        axs[0].plot(listk, revenue_bayes_ucb_degree, label = 'Bayesian UCB')
        axs[0].plot(listk, revenue_thompson_sampling_degree, label = 'Thompson Sampling')
        axs[0].set_xlabel('k')
        axs[0].set_xticks(listk, listk)
        axs[0].set_ylabel('Revenue')
        axs[0].set_title("Degree")
        axs[0].legend()

        axs[1].plot(listk, revenue_ucb_page, label = 'UCB')
        axs[1].plot(listk, revenue_bayes_ucb_page, label = 'Bayesian UCB')
        axs[1].plot(listk, revenue_thompson_sampling_page, label = 'Thompson Sampling')
        axs[1].set_xlabel('k')
        axs[1].set_xticks(listk, listk)
        axs[1].set_ylabel('Revenue')
        axs[1].set_title("PageRank")
        axs[1].legend()


        axs[2].plot(listk, revenue_ucb_vote, label = 'UCB')
        axs[2].plot(listk, revenue_bayes_ucb_vote, label = 'Bayesian UCB')
        axs[2].plot(listk, revenue_thompson_sampling_vote, label = 'Thompson Sampling')
        axs[2].set_xlabel('k')
        axs[2].set_xticks(listk, listk)
        axs[2].set_ylabel('Revenue')
        axs[2].set_title("VoteRank")
        axs[2].legend()

        axs[3].plot(listk, revenue_ucb_s_deg, label = 'UCB')
        axs[3].plot(listk, revenue_bayes_ucb_s_deg, label = 'Bayesian UCB')
        axs[3].plot(listk, revenue_thompson_sampling_s_deg, label = 'Thompson Sampling')
        axs[3].set_xlabel('k')
        axs[3].set_xticks(listk, listk)
        axs[3].set_ylabel('Revenue')
        axs[3].set_title("shapley degree")
        axs[3].legend()

        axs[4].plot(listk, revenue_ucb_s_thresh, label = 'UCB')
        axs[4].plot(listk, revenue_bayes_ucb_s_thresh, label = 'Bayesian UCB')
        axs[4].plot(listk, revenue_thompson_sampling_s_thresh, label = 'Thompson Sampling')
        axs[4].set_xlabel('k')
        axs[4].set_xticks(listk, listk)
        axs[4].set_ylabel('Revenue')
        axs[4].set_title("shapley threshold")
        axs[4].legend()

        axs[5].plot(listk, revenue_ucb_s_clos, label = 'UCB')
        axs[5].plot(listk, revenue_bayes_ucb_s_clos, label = 'Bayesian UCB')
        axs[5].plot(listk, revenue_thompson_sampling_s_clos, label = 'Thompson Sampling')
        axs[5].set_xlabel('k')
        axs[5].set_xticks(listk, listk)
        axs[5].set_ylabel('Revenue')
        axs[5].set_title("shapley closeness")
        axs[5].legend()

        fig.tight_layout(pad=0.5)
        plt.savefig(path+'armset_chart.png')
        #plt.show()"""

        


        

        if auction =="MUDAN":
            print(revenue_bayes_ucb_degree)
            axs[0].plot(listk, revenue_ucb_degree, label = 'Degree')
            axs[0].plot(listk, revenue_ucb_page, label = 'PageRank')
            axs[0].plot(listk, revenue_ucb_vote, label = 'VoteRank')
            axs[0].plot(listk, revenue_ucb_normal, label = 'Normal')
            axs[0].plot(listk, revenue_ucb_s_deg, label = 's. degree')
            axs[0].plot(listk, revenue_ucb_s_thresh, label = 's. threshold')
            axs[0].plot(listk, revenue_ucb_s_clos, label = 's. closeness')
            axs[0].set_xlabel('k')
            axs[0].set_xticks(listk, listk)
            axs[0].set_ylabel('Revenue')
            axs[0].set_title("UCB")
            axs[0].legend()
            

        else:
            print(revenue_ucb_degree)
            axs[1].plot(listk, revenue_ucb_degree, label = 'Degree')
            axs[1].plot(listk, revenue_ucb_page, label = 'PageRank')
            axs[1].plot(listk, revenue_ucb_vote, label = 'VoteRank')
            axs[1].plot(listk, revenue_ucb_normal, label = 'Normal')
            axs[1].plot(listk, revenue_ucb_s_deg, label = 's. degree')
            axs[1].plot(listk, revenue_ucb_s_thresh, label = 's. threshold')
            axs[1].plot(listk, revenue_ucb_s_clos, label = 's. closeness')
            axs[1].set_xlabel('k')
            axs[1].set_xticks(listk, listk)
            axs[1].set_ylabel('Revenue')
            axs[1].set_title("MUDAN (UCB)")
            axs[1].legend()
            
        """

        axs[2].plot(listk, revenue_thompson_sampling_degree, label = 'Degree')
        axs[2].plot(listk, revenue_thompson_sampling_page, label = 'PageRank')
        axs[2].plot(listk, revenue_thompson_sampling_vote, label = 'VoteRank')
        axs[2].plot(listk, revenue_thompson_sampling_normal, label = 'Normal')
        axs[2].plot(listk, revenue_thompson_sampling_s_deg, label = 's. degree')
        axs[2].plot(listk, revenue_thompson_sampling_s_thresh, label = 's. threshold')
        axs[2].plot(listk, revenue_thompson_sampling_s_clos, label = 's. closeness')
        axs[2].set_xlabel('k')
        axs[2].set_xticks(listk, listk)
        axs[2].set_ylabel('Revenue')
        axs[2].set_title("Thompson Sampling")
        axs[2].legend()"""

    fig.tight_layout(pad=0.5)
    if not os.path.exists(path+'/plot/'):
        os.makedirs(path+'/plot/')
    plt.savefig(path+'/plot/'+auction+'_bandit_charts_2.png')

    print(listk)
        #plt.show()


    print(df_page_k.reset_index(drop=True))