# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 17:17:18 2022

@author: Nour El Houda Hammami
"""



import pandas as pd
import pickle

# Assuming your PPO dictionary of simulations is named "data"
data = open("./dico_data_multi.pkl", "rb")
data = data.read()
data = pickle.loads(data)

# Assuming your CPLEX dictionary of simulations is named "cplex_data"
cplex_data = open("./cplex_data.pkl", "rb")
cplex_data = cplex_data.read()
cplex_data = pickle.loads(cplex_data)

mip_data = open("./MIP_data.pkl", "rb")
mip_data = mip_data.read()
mip_data = pickle.loads(mip_data)

# Creating empty lists to store the extracted data
simulations = []
hours = []
ms_values = []
de_values = []
added_values = []
compute_values = []

# Looping over simulations and extracting the required metrics
for simulation in data:
    for hour in data[simulation]["Ms"]:
        ms = data[simulation]["Ms"][hour]
        de = data[simulation]["DE"][hour]
        added = data[simulation]["added"][hour]
        compute =data[simulation]["compute_time"][hour]
        
        # Appending the extracted data to the respective lists
        simulations.append(simulation)
        hours.append(hour)
        ms_values.append(ms)
        de_values.append(de)
        added_values.append(added)
        compute_values.append(compute)

# Creating a pandas DataFrame from the extracted data
df = pd.DataFrame({
    "Simulation": simulations,
    "Hour": hours,
    "ms": ms_values,
    "de": de_values,
    "added": added_values,
    "compute_time": compute_values

})

filtered_df = df[df["added"] != 0]

# Creating empty lists to store the extracted CPLEX data
cplex_simulations = []
cplex_hours = []
cplex_ms_values = []
cplex_de_values = []
cplex_added_values = []
cplex_compute_values= []

# Looping over simulations and extracting the required metrics from the CPLEX dictionary
for simulation in cplex_data:
    for hour in cplex_data[simulation]["Ms_cp"]:
        ms = cplex_data[simulation]["Ms_cp"][hour]
        de = cplex_data[simulation]["DE_cp"][hour]
        added = cplex_data[simulation]["added_cp"][hour]
        compute_cp = cplex_data[simulation]["compute_time_cp"][hour]

        # Appending the extracted data to the respective lists
        cplex_simulations.append(simulation)
        cplex_hours.append(hour)
        cplex_ms_values.append(ms)
        cplex_de_values.append(de)
        cplex_added_values.append(added)
        cplex_compute_values.append(compute_cp)

# Creating a pandas DataFrame from the extracted CPLEX data
cplex_df = pd.DataFrame({
    "Simulation": cplex_simulations,
    "Hour": cplex_hours,
    "ms_cplex": cplex_ms_values,
    "de_cplex": cplex_de_values,
    "added_cplex": cplex_added_values,
    "compute_cplex": cplex_compute_values

})

filtered_cplex=cplex_df[cplex_df["added_cplex"] != 0]

# Creating empty lists to store the extracted MIP data
mip_simulations = []
mip_hours = []
mip_ms_values = []
mip_de_values = []
mip_added_values = []
mip_compute_values= []

# Looping over simulations and extracting the required metrics from the mip dictionary
for simulation in mip_data:
    for hour in mip_data[simulation]["Ms_cp"]:
        ms = mip_data[simulation]["Ms_cp"][hour]
        de = mip_data[simulation]["DE_cp"][hour]
        added = mip_data[simulation]["added_cp"][hour]
        compute_cp = mip_data[simulation]["compute_time_cp"][hour]

        # Appending the extracted data to the respective lists
        mip_simulations.append(simulation)
        mip_hours.append(hour)
        mip_ms_values.append(ms)
        mip_de_values.append(de)
        mip_added_values.append(added)
        mip_compute_values.append(compute_cp)

# Creating a pandas DataFrame from the extracted mip data
mip_df = pd.DataFrame({
    "Simulation": mip_simulations,
    "Hour": mip_hours,
    "ms_mip": mip_ms_values,
    "de_mip": mip_de_values,
    "added_mip": mip_added_values,
    "compute_mip": mip_compute_values

})

filtered_mip=mip_df[mip_df["added_mip"] != 0]

# Performing statistical calculations on the DataFrames
mean_ms = filtered_df["ms"].mean()
# print("Mean ms:", mean_ms)
mean_de = filtered_df["de"].mean()
mean_time= filtered_df["compute_time"].mean()

mean_ms_cp = filtered_cplex["ms_cplex"].mean()
# print("Mean ms cp:", mean_ms_cp)
mean_de_cp = filtered_cplex["de_cplex"].mean()
mean_time_cp= filtered_cplex["compute_cplex"].mean()

mean_ms_mip = filtered_mip["ms_mip"].mean()
# print("Mean ms mip:", mean_ms_mip)
mean_de_mip = filtered_mip["de_mip"].mean()
mean_time_mip= filtered_mip["compute_mip"].mean()

# Print the statistical results
print("Mean de:", mean_de ,mean_de_cp, mean_de_mip )
print("mean time:", mean_time,mean_time_cp, mean_time_mip)
obj_ppo=0.50 * mean_ms + 0.50 * mean_de
obj_cp=0.50 * mean_ms_cp + 0.50 * mean_de_cp
obj_mip=0.50 * mean_ms_mip + 0.50 * mean_de_mip
print("obj_ppo,obj_cp,obj_mip:", obj_ppo,obj_cp,obj_mip)
gap_cp=( (0.50* mean_ms + 0.50 * mean_de) / (0.50* mean_ms_cp + 0.50 * mean_de_cp) - 1)*100
gap_mip=( (0.50* mean_ms + 0.50 * mean_de) / (0.50* mean_ms_mip + 0.50 * mean_de_mip) - 1)*100
comparison_df = pd.DataFrame({
    "Method": ["PPO", "CPLEX" ,"MIP","Gap_cp(%)", "Gap_mip(%)"],
    "Mean_ms": [mean_ms, mean_ms_cp,mean_ms_mip,((mean_ms-mean_ms_cp) / mean_ms_cp)*100,( (mean_ms-mean_ms_mip) / mean_ms_mip)*100],
    "Mean_deviation": [mean_de, mean_de_cp,mean_de_mip, '-','-'],
    "Mean_computational_time":[mean_time,mean_time_cp,mean_time_mip,"-","-"],
    "Objective_function":[obj_ppo ,obj_cp ,obj_mip,gap_cp,gap_mip]
    
})
gap=(gap_cp,gap_mip)

print("gap is:", gap )

comparison_df.to_excel("./comparison_results_mp_time.xlsx", index=False)
