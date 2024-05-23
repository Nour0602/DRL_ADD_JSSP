# DRL_ADD_JSSP
The repository contains the codes used for generating experimental results of the article: "Design and calibration of a DRL algorithm for solving the job shop scheduling problem under unexpected job arrivals".

Artcile DOI: https://doi.org/10.1007/s10696-024-09540-2

For citing the article, please use the following citation:
Hammami, N.E.H., Lardeux, B., B. Hadj-Alouane, A. et al. Design and calibration of a DRL algorithm for solving the job shop scheduling problem under unexpected job arrivals. Flex Serv Manuf J (2024). https://doi.org/10.1007/s10696-024-09540-2

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The codes are used for addressing  a real-time  JSSP with unexpected job arrivals using PPO-AC algorithm, a policy gradient RL algorithm, combined with a GNN architecture.
The optimization objective is the minimization of the weighted sum of makespan (total completion time of operations) and jobs total deviation,
refrring to respectively efficiency and stability criteria. 

For reproducing test results of the proposed PPO-AC and generate 
comparison results with CPOPTIMIZER and MIP models:

     Run "Store-sim" for generating instance simulations.
     Run "Test-Performance-PPO.py" for solving simulations using PPO-AC algorithm.
     Run "CPOPT" for solving simulations using CPOPTIMIZER.
     Run "MIP" for solving simulations using MIP.
     Run "Compare-res" to generate comparison results.

Test results can be generated for different values of alpha parameter (the deviation weight parameter).
Models correspending to different alpha values are located in "alpha_analysis_models".
