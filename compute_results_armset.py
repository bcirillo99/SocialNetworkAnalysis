import os
import pandas as pd

for t in ["20000","40000","60000"]:
    df_mudan = pd.read_csv("results_armset/"+t+"/MUDAN_2.csv")
    df_mudar = pd.read_csv("results_armset/"+t+"/MUDAR_2.csv")
    #df_vcg = pd.read_csv("final_results/20000/VCG_2.csv")
    df_gidm = pd.read_csv("results_armset/"+t+"/GIDM_2.csv")

    """df_mudan = df.loc[df['Auction'] == "MUDAN"].reset_index(drop=True)
    df_mudar = df.loc[df['Auction'] == "MUDAR"].reset_index(drop=True)
    df_mixed = df.loc[df['Auction'] == "MIXED"].reset_index(drop=True)"""

    results = []
    listk = [2,3,4,5]



    for k in listk:
        
        df_mudan_k = df_mudan.iloc[[df_mudan.loc[df_mudan['k'] == k]['Revenue'].idxmax()]]
        df_mudar_k = df_mudar.iloc[[df_mudar.loc[df_mudar['k'] == k]['Revenue'].idxmax()]]
        #df_vcg_k = df_vcg.iloc[[df_vcg.loc[df_vcg['k'] == k]['Revenue'].idxmax()]]
        df_gidm_k = df_gidm.iloc[[df_gidm.loc[df_gidm['k'] == k]['Revenue'].idxmax()]]
        results.append(pd.concat([df_mudan_k,df_mudar_k,df_gidm_k]))
        #results.append(pd.concat([df_gidm_k]))

    final_results = pd.concat(results).reset_index(drop=True)
    #print(final_results)
    path = "results_armset/"+t+"/results/"
            
    if not os.path.exists(path):
        os.makedirs(path)
    final_results.to_csv(path+"basic_auction_best_results_2.csv")

    results = []
    for k in listk:
        df_k = final_results.iloc[[final_results.loc[final_results['k'] == k]['Revenue'].idxmax()]]
        results.append(df_k)

    final_results = pd.concat(results).reset_index(drop=True)
    print(final_results)
    #final_results.to_csv(path+"best_results_2.csv",columns=["Bandit", "T",  "k", "Revenue", "armset"])
    final_results.to_csv(path+"best_results_2.csv",columns=["Bandit", "T",  "k", "Revenue", "armset"])