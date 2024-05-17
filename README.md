# DRL_ADD_JSSP
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
  Models correspending to different alpha values are located in "alpha_analysis_models"
