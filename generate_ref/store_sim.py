# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 17:17:18 2022

@author: Nour El Houda Hammami
"""
import numpy as np  
import pickle 
from uniform_instance_gen_re import uni_instance_gen
from generate_ref_cp import*


N_JOBS_N =2
print(  N_JOBS_N)
N_MACHINES_N = 2
print(N_MACHINES_N)
data_generator = uni_instance_gen

list_store_sim={}
N_sim=25
for simulation in range(N_sim):
     
     dur,seq=data_generator(n_j=N_JOBS_N, n_m=N_MACHINES_N, low=1, high=99)
     data1,data2,start,mchIds,end, sol=optimize_and_visualize(N_MACHINES_N,N_JOBS_N,dur,seq, 1, 1)
     data=(data1,data2,mchIds,start,end,sol)
     list_store_sim[simulation]=data
list_store_file=open("./store_data.pkl", "wb")
pickle.dump(list_store_sim, list_store_file)
list_store_file.close() 