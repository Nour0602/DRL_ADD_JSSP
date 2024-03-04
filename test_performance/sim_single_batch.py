
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 17:17:18 2022

@author: Nour El Houda Hammami
"""
import numpy as np
import torch
import time
import pickle

import argparse
from Params_re import configs
from uniform_instance_gen_re import uni_instance_gen
from mb_agg_re import *
from agent_utils_re import *
from JSSP_Env_re import SJSSP 
from early import PPO
from generate_ref_cp import*
np.set_printoptions(suppress=True)

#Configuration
device = torch.device(configs.device)
parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')  
parser.add_argument('--Nn_j', type=int, default=15, help='Number of jobs on which to be loaded net are trained')
parser.add_argument('--Nn_m', type=int, default=15, help='Number of machines on which to be loaded net are trained')
parser.add_argument('--seed', type=int, default=200, help='Seed for validate set generation')   
params_re = parser.parse_args()

#Principal tetsing 
#Prepare Dimension/Data
#Define Number of jobs/machines of the instance
N_JOBS_N =15
print(  N_JOBS_N)
N_MACHINES_N = 15
print(N_MACHINES_N)

#Processing time interval consideration
LOW = 1
HIGH = 99                                                                                     
SEED = params_re.seed
batch=1

# load optimal policy
path = './50/{}.pth'.format(str(15) + '_' + str(115) + '_' + str(LOW) + '_' + str(HIGH))
data_generator = uni_instance_gen
# define calculations per simulation: 
dico_data_multi={}
list_store_sim={}
data_dict = open("./store_data.pkl", "rb")
data_dict = data_dict.read()
data_dict = pickle.loads(data_dict)

N_sim=len(data_dict)

for simulation in range(N_sim):
    print("simulation is:", simulation)
    # #Prepare simulation data
    (data1,data2,mchIds,start,end,sol)=data_dict[simulation]
    # print("mchIds",mchIds)
    # print("data2", data2)
    data=(data1,data2,mchIds,start,end,sol)
    M=np.amax(end)

    # definition of start time on machines
    start_operation=np.zeros_like(data1, dtype=np.int32)
    for op in range((data1.shape[0])*N_MACHINES_N):
                
                mch_op=np.take(data2,op)-1
                ind1=np.where(mchIds[mch_op]==op)[0][0]
                row_op=op // N_MACHINES_N
                col_op=op % N_MACHINES_N 
                start_operation[row_op,col_op]= start[mch_op,ind1]
            
    dico_dur, dico_mchs_seq, dico_mchIds, dico_start, dico_end, dico_seq, dico_added, dico_eliminated,dico_start_operation, dico_ms, dico_de, dico_compute_time={},{},{},{},{},{},{},{},{}, {},{},{}


    dico_ms[0]= int(end.max())
    dico_de[0]=0

    dico_dur[0]=data1
    dico_mchs_seq[0]=data2
    dico_mchIds[0]=mchIds
    dico_start[0]=start
    dico_start_operation[0]=start_operation
    dico_end[0]=end
    dico_seq[0]=sol
    dico_added[0]=0
    dico_eliminated[0]=[]
    dico_compute_time[0]=0

    executed_ops=[]



    hour=1
    tresh=int(0.9 *dico_ms[0])


    while hour<=tresh:
        executed_ops=[]
        if  hour==tresh:  
            # define batch of new jobs   
            add_poisson=3
     
        else:
            add_poisson=0              
        dico_added[hour]=add_poisson

        N_j_t = N_JOBS_N
        for key in dico_added:
            N_j_t +=dico_added[key]

        N_m_t = N_MACHINES_N
        #define the environment
        env = SJSSP(n_j=N_j_t, n_m=N_m_t)
        ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
                    n_j = N_j_t,
                    n_m = N_m_t,
                    num_layers=configs.num_layers,
                    neighbor_pooling_type=configs.neighbor_pooling_type,
                    input_dim=configs.input_dim,
                    hidden_dim=configs.hidden_dim,
                    num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                    num_mlp_layers_actor=configs.num_mlp_layers_actor,
                    hidden_dim_actor=configs.hidden_dim_actor,
                    num_mlp_layers_critic=configs.num_mlp_layers_critic,
                    hidden_dim_critic=configs.hidden_dim_critic)

        ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu'))) 
        data_add=data_generator(n_j=add_poisson, n_m=N_MACHINES_N, low=1, high=99) #data1,data2,mchIds,start,end,sol
            
            
        if add_poisson!=0:
            # definition of already executed/being executed operations:   
            for op in range((dico_dur[hour-1].shape[0])*N_m_t):
                
                mch_op=np.take(dico_mchs_seq[hour-1],op)-1
                
                ind1=np.where(dico_mchIds[hour-1][mch_op]==op)[0][0]
                row_op=op // N_MACHINES_N
                col_op=op % N_MACHINES_N 

                if ((dico_end[hour-1][row_op,col_op]>hour and dico_start_operation[hour-1][row_op,col_op]<hour) or (dico_end[hour-1][row_op,col_op]<=hour)):
                        executed_ops. append(op)
            dico_eliminated[hour]=executed_ops
            
            adj, dev,fea, candidate,candidate_ch,mask, nb_nodes, temp1, opIDsOnMchs,mchsStartTimes = env.reset_add_eliminate((dico_dur[hour-1],dico_mchs_seq[hour-1]), dico_start[hour-1],dico_mchIds[hour-1],add_poisson,data_add,[executed_ops[i] for i in range(len(executed_ops))],True,dico_end[hour-1],dico_end[0])

            g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                                            batch_size=torch.Size([ 1, env.number_of_tasks-len(executed_ops), env.number_of_tasks-len(executed_ops)]),
                                            n_nodes=env.number_of_tasks-len(executed_ops),
                                            device=device) 
            starting= time.time()
            ep_reward = -env.max_endTime
          
            DEV=[]
            while True:
                        fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
                        adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
                        candidate_tensor = torch.from_numpy(np.copy(candidate_ch)).to(device)
                        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
                        with torch.no_grad():
                            pi,v = ppo.policy(x=fea_tensor.float(),
                                                graph_pool=g_pool_step,
                                                padded_nei=None,
                                                adj=adj_tensor,
                                                candidate=candidate_tensor.unsqueeze(0),
                                                mask=mask_tensor.unsqueeze(0))
                                # action = sample_select_action(pi, candidate)
                        action_idk = greedy_select_action(pi, candidate_ch)
                        key_listed=list(env.idd.keys())
                        value_listed=list(env.idd.values())
                        
                        position2=key_listed.index(action_idk)
                        action_idv=value_listed[position2]
                                        
                        action_idk=np.int64(action_idk)
                        action_idv=np.int64(action_idv)            
                        adj,dev, fea, reward, done, candidate_ch, mask= env.step(hour,N_JOBS_N,action_idk.item(),action_idv.item(), executed_ops,add_poisson) 
                   
                        DEV.append(dev)
                        
                        ep_reward += reward
                        if done:
                            break
            ending=time.time()
            dico_compute_time[hour]=ending-starting
                        
            sum1 = DEV[-1][:,-1].sum() 
            #  Define the start-operation matrix 
            start_operation=np.zeros_like(env.dur, dtype=np.single)
            for op in range(env.dur.shape[0]*N_MACHINES_N): 
                
                mch_op=np.take(env.m,op)-1
                ind1=np.where(env.opIDsOnMchs[mch_op]==op)[0][0]
                row_op=op // N_MACHINES_N
                col_op=op % N_MACHINES_N 
                start_operation[row_op,col_op]= env.mchsStartTimes[mch_op,ind1]
            
            # Affect data of the hour
            dico_dur[hour]=env.dur
            dico_mchs_seq[hour]=env.m
            dico_mchIds[hour]=env.opIDsOnMchs
            dico_start[hour]=env.mchsStartTimes
            dico_start_operation[hour]=start_operation
            dico_end[hour]=env.temp1
            dico_seq[hour]=env.partial_sol_sequeence
            dico_ms[hour]=dico_end[hour].max()
            dico_de[hour]=sum1
        
        else:
            
            dico_eliminated[hour]= dico_eliminated[hour-1]       
            dico_dur[hour]=dico_dur[hour-1]
            dico_mchs_seq[hour]=dico_mchs_seq[hour-1]
            dico_mchIds[hour]=dico_mchIds[hour-1]
            dico_start[hour]=dico_start[hour-1]
            dico_start_operation[hour]=dico_start_operation[hour-1]
            dico_end[hour]=dico_end[hour-1]
            dico_seq[hour]=dico_seq[hour-1] 

            dico_ms[hour]=dico_ms[hour-1]
            dico_de[hour]=0
            dico_compute_time[hour]=0
        # Make data as int.arrays for the exact method
        for hour in dico_dur:
            dico_dur[hour] =  dico_dur[hour].astype(int)
            dico_mchs_seq[hour]=dico_mchs_seq[hour].astype(int)
            dico_mchIds[hour]=dico_mchIds[hour].astype(int)
            dico_start[hour]=dico_start[hour].astype(int)
            dico_start_operation[hour]=dico_start_operation[hour].astype(int)
            dico_end[hour]=dico_end[hour].astype(int)
        hour+=1
    dico_sim={}
    dico_sim["Ms"]=dico_ms
    dico_sim["DE"]=dico_de
    dico_sim["compute_time"]=dico_compute_time
    dico_sim["added"]=dico_added
    dico_sim["elimiminated_ops"]=dico_eliminated
    dico_sim["dur"]=dico_dur
    dico_sim["ops_seq"]=dico_mchs_seq
    dico_sim["mchstart_times"]=dico_start
    dico_sim["mchIds"]=dico_mchIds
    dico_sim["op_start"]=dico_start_operation
    dico_sim["End"]=dico_end
    dico_sim["sol_seq"]=dico_seq
    

    dico_data_multi[simulation]=dico_sim 
        
scenarios_file=open("./50/3/dico_data_multi.pkl", "wb")
pickle.dump(dico_data_multi, scenarios_file)
scenarios_file.close()   

print("calculation is over!")
























