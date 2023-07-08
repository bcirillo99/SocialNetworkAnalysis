# Social Network Analysis project
The repo contains code and report of the project for the university exam *Social Network Analysis*. This project is divided in two part (midterm and final project) and the goal of the various tasks are explained in the *midterm.pdf* and *final.pdf* files contained in the **outlines** folder.

The final project is composed of two task:

## Task1: Network Identification
The goal of this task is to analyze the network **net_4** with the network mining tools and guess which model has been used for creating it. The guess has to be supported by an appropriate set of experiments to confirm that networks generated with the proposed model have characteristics similar to N (note that the goal is also to guess also the parameters of the model).

## Task2: Mechanisms on Weighted Social Networks with Unknown Weights
The goal of this task is to find in a social network, through the use of the **bandit algorithms** and **multi-unit auctions**, at each step the node (seller) that maximizes total revenue (check the *final.pdf* file for more details).

## Repository structure

The repo is structured as follows:
- There are 4 folders named **task1**, **task2**, **task3** and **task4** containing codes and results for midterm project tasks.
- The folder **task1_final** containing codes and results for the task 1 of the final project
- The folder **results** containing the results of the task 2 of the final profect
- The python file *SocNetMec.py* contains the class required for task 2 of the final profect
- The python file *final_test.py* contains the code for running the task 2 of the final profect
- The python file *model.py* contains functions to generate different types of social network models
- There are also 2 pdf file *task1_report.pdf* and *Report_Social_Network_Analysis.pdf* where they explain the various decisions and analyses that led to these results:

For the **task 1** we found that the model used to generate the **net_4** network is the **Watts-Strogatz** with the following parameters:
- r = 2.71
- k = 1
- q = 4

For **task 2**, the bandit algorithm used was **UCB1** with the use of a reduced armset containing the 200 nodes with the higher pageRank, while the auction format chosen was **GIDM**.