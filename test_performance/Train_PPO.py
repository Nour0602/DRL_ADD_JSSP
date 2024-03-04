# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 23:24:42 2022

@author: ISEN
"""
from mb_agg_re import *
from agent_utils_re import eval_actions
from agent_utils_re import select_action
from actor_critic_re import ActorCritic
from copy import deepcopy
import torch
import time
import torch.nn as nn
import numpy as np
from Params_re import configs
from generate_ref_cp import*
# from validation_re import validate_1
from statistics import mean
device = torch.device(configs.device)
# import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.logprobs = []

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.logprobs[:]


class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 n_j,
                 n_m,
                 num_layers,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(n_j=n_j,
                                  n_m=n_m,
                                  num_layers=num_layers,
                                  learn_eps=False,
                                  neighbor_pooling_type=neighbor_pooling_type,
                                  input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                  num_mlp_layers_actor=num_mlp_layers_actor,
                                  hidden_dim_actor=hidden_dim_actor,
                                  num_mlp_layers_critic=num_mlp_layers_critic,
                                  hidden_dim_critic=hidden_dim_critic,
                                  device=device)
        self.policy_old = deepcopy(self.policy)

        '''self.policy.load_state_dict(
            torch.load(path='./{}.pth'.format(str(n_j) + '_' + str(n_m) + '_' + str(1) + '_' + str(99))))'''

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)

        self.V_loss_2 = nn.MSELoss()

    def update(self, memories, n_tasks, g_pool):

        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef

        rewards_all_env = []
        adj_mb_t_all_env = []
        fea_mb_t_all_env = []
        candidate_mb_t_all_env = []
        mask_mb_t_all_env = []
        a_mb_t_all_env = []
        old_logprobs_mb_t_all_env = []
        # store data for all env
        for i in range(len(memories)):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memories[i].r_mb), reversed(memories[i].done_mb)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_all_env.append(rewards)
            # process each env data
            adj_mb_t_all_env.append(aggr_obs(torch.stack(memories[i].adj_mb).to(device), n_tasks))
            fea_mb_t = torch.stack(memories[i].fea_mb).to(device)
            fea_mb_t = fea_mb_t.reshape(-1, fea_mb_t.size(-1))
            fea_mb_t_all_env.append(fea_mb_t)
            candidate_mb_t_all_env.append(torch.stack(memories[i].candidate_mb).to(device).squeeze())
            mask_mb_t_all_env.append(torch.stack(memories[i].mask_mb).to(device).squeeze())
            a_mb_t_all_env.append(torch.stack(memories[i].a_mb).to(device).squeeze())
            old_logprobs_mb_t_all_env.append(torch.stack(memories[i].logprobs).to(device).squeeze().detach())

        # get batch argument for net forwarding: mb_g_pool is same for all env
        mb_g_pool = g_pool_cal(g_pool, torch.stack(memories[0].adj_mb).to(device).shape, n_tasks, device)

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            ploss_sum = 0
            for i in range(len(memories)):
                pis, vals = self.policy(x=fea_mb_t_all_env[i].float(),
                                        graph_pool=mb_g_pool,
                                        adj=adj_mb_t_all_env[i],
                                        candidate=candidate_mb_t_all_env[i],
                                        mask=mask_mb_t_all_env[i],
                                        padded_nei=None)
                logprobs, ent_loss = eval_actions(pis.squeeze(), a_mb_t_all_env[i])
                ratios = torch.exp(logprobs - old_logprobs_mb_t_all_env[i].detach())
                advantages = rewards_all_env[i] - vals.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                v_loss = self.V_loss_2(vals.squeeze(), rewards_all_env[i])
                p_loss = - torch.min(surr1, surr2)
                ent_loss = - ent_loss.clone()
                loss = vloss_coef * v_loss + ploss_coef * p_loss + entloss_coef * ent_loss
                loss_sum += loss
                vloss_sum += v_loss
                ploss_sum += p_loss
                ploss_sum = -1*ploss_sum 
            self.optimizer.zero_grad()
            loss_sum.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        if configs.decayflag:
            self.scheduler.step()
        return loss_sum.mean().item(), vloss_sum.mean().item(), ploss_sum.mean().item()


def main():

    from JSSP_Env_re import SJSSP
    envs = [SJSSP(n_j=configs.n_j, n_m=configs.n_m) for _ in range(configs.num_envs)]
    
    
    from uniform_instance_gen_re import uni_instance_gen
    data_generator = uni_instance_gen
   
    from generate_ref import test
 

    
    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)    
    np.random.seed(configs.np_seed_train)
    
    add_poisson=1
     
    memories = [Memory() for _ in range(configs.num_envs)]
    
    ppo_1 = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
            n_j=envs[0].number_of_jobs+add_poisson,
            n_m=envs[0].number_of_machines, 
            num_layers=configs.num_layers,
            neighbor_pooling_type=configs.neighbor_pooling_type,
            input_dim=configs.input_dim,
            hidden_dim=configs.hidden_dim,
            num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
            num_mlp_layers_actor=configs.num_mlp_layers_actor,
            hidden_dim_actor=configs.hidden_dim_actor,
            num_mlp_layers_critic=configs.num_mlp_layers_critic,
            hidden_dim_critic=configs.hidden_dim_critic)
           
    g_pool_step_1=g_pool_cal(graph_pool_type=configs.graph_pool_type, #nb of nodes msut be generalized for all environments
                             batch_size=torch.Size([1, envs[0].number_of_tasks+add_poisson*envs[0].number_of_machines-45,envs[0].number_of_tasks+add_poisson*envs[0].number_of_machines-45]),
                             n_nodes=envs[0].number_of_tasks+add_poisson*envs[0].number_of_machines-45,
                             device=device)
    

    # training loop
    score_m=[]
    lo=[]
    lop=[]
    lo1=[]
    log = []

     
    # validation_log_ms = []
    # validation_log_dev = []
    # optimal_gaps = []
    # optimal_gap = 1
    # record_ms = 10000
    # record_dev=10000

    reward_list=[]
    ms_list=[]
    de_list=[]
    sum_list=[]

    moving_sum=[]
    moving_avg = []
    moving_de =[]
    moving_ms=[] 


    delta=10
    window_size=10000
    i_update=0

    STOP=False
    while STOP==False:
        
              
        ep_rewards = [0 for _ in range(configs.num_envs)]
        ep_ms=[0 for _ in range(configs.num_envs)]
        ep_de=[0 for _ in range(configs.num_envs)]
       
        tuple_cmax=[0,0] 

        adj_envs = []
        fea_envs = []
        candidate_envs = []
        mask_envs = []
        DEV=[]
        data=[]
       
        for i, env in enumerate(envs):
            dur,seq=data_generator(n_j=configs.n_j, n_m=configs.n_m, low=1, high=99)
            data_add=data_generator(n_j=add_poisson, n_m=configs.n_m, low=1, high=99)
            # data1,data2,mchIds,start,end, sol_seq=test()
            data1,data2,start,mchIds,end, sol_seq=optimize_and_visualize(configs.n_m,configs.n_j,dur,seq, 0.1, 1)
            # print(  "End at time ref:", end)
            tuple_cmax[i]=end.max()
            data=(data1,data2)
           
            eliminated=sol_seq
          
            while (len(eliminated))>45:
                eliminated.pop()
            # print("eliminated are:", eliminated)
            row_eli=eliminated[-1]//configs.n_m
            col_eli=eliminated[-1]  % configs.n_m
            hour=end[row_eli,col_eli]
            
            # print("end is :",end)
            # print("hour is :",hour)

            adj, dev,fea, candidate,candidate_ch,mask, nb_nodes, temp1, opIDsOnMchs,mchsStartTimes = env.reset_add_eliminate(data,start,mchIds,add_poisson,data_add,[eliminated[i]for i in range(len(eliminated))],True,end,end) #

            adj_envs.append(adj)
            fea_envs.append(fea)
            candidate_envs.append(candidate_ch)
            mask_envs.append(mask)
            ep_rewards[i] = - env.initQuality

            
      
        while True:
            fea_tensor_envs = [torch.from_numpy(np.copy(fea)).to(device) for fea in fea_envs]
            adj_tensor_envs = [torch.from_numpy(np.copy(adj)).to(device).to_sparse() for adj in adj_envs]
            candidate_tensor_envs = [torch.from_numpy(np.copy(candidate)).to(device) for candidate in candidate_envs]
            mask_tensor_envs = [torch.from_numpy(np.copy(mask)).to(device) for mask in mask_envs]
                       
            with torch.no_grad():
                action_envs = []
                a_idx_envs = []
                for i in range(configs.num_envs):
                    pi1, _ = ppo_1.policy_old(x=fea_tensor_envs[i].float(),
                                           graph_pool=g_pool_step_1,
                                           padded_nei=None,
                                           adj=adj_tensor_envs[i],
                                           candidate=candidate_tensor_envs[i].type(torch.int64).unsqueeze(0),
                                           mask=mask_tensor_envs[i].unsqueeze(0))

                    action_idk, a_idx = select_action(pi1, candidate_envs[i], memories[i])
                    
                    key_listed=list(envs[i].idd.keys())
                    value_listed=list(envs[i].idd.values())
                    
                    #print("key_listed,value_listed at training:", key_listed,value_listed)
                    
                    position2=key_listed.index(action_idk)
                    action_idv=value_listed[position2]
                    

                    action_idk=np.int64(action_idk)
                    action_idv=np.int64(action_idv)
                    action_envs.append([action_idk,action_idv])
                    
                    a_idx_envs.append(a_idx)
            
            
            adj_envs = []
            fea_envs = []
            candidate_envs = []
            mask_envs = []
            
            # Saving episode data
            DEV=[]
            for i in range(configs.num_envs):
                memories[i].adj_mb.append(adj_tensor_envs[i])
                memories[i].fea_mb.append(fea_tensor_envs[i])
                memories[i].candidate_mb.append(candidate_tensor_envs[i])
                memories[i].mask_mb.append(mask_tensor_envs[i])
                memories[i].a_mb.append(a_idx_envs[i])

                adj,dev, fea, reward, done, candidate_ch, mask = envs[i].step(hour,configs.n_j,action_envs[i][0].item(),action_envs[i][1].item(),[eliminated[i]for i in range(len(eliminated))],add_poisson)
                        
                adj_envs.append(adj)
                DEV.append(dev) ##deviation matrix 
                fea_envs.append(fea)
                candidate_envs.append(candidate_ch)
                mask_envs.append(mask)
                
                ep_rewards[i] += reward
                # sum1=dev[:,-1].sum()
                # sum1=np.max(dev)
                # print("sum1", sum1)
                # print("dev per step is:", dev)
                
                # ep_de[i]=sum1
                
                # print("ep_de[i] de env i**", ep_de[i], i)
                memories[i].r_mb.append(reward)
                memories[i].done_mb.append(done)

               
            if envs[0].done():
                break
            
        for j in range(configs.num_envs):
            ep_rewards[j] -= envs[j].posRewards
            ep_ms[j]=envs[j].temp1.max()
            ep_de[j]=np.max(envs[j].dev)
            
            # print("ep de:", ep_de[0])
            # print("envs[j].temp1:", envs[j].temp1)
            # print("ep_ms[j]:", ep_ms[j])
            # print("reward is:", ep_rewards[j])
            # print("ep_de[i] de env i", ep_de[i], i)
            # print("ep_ms[j], envs[j].temp1.max():", ep_ms[j], envs[j].temp1.max())
           
        
        
      
       
        loss, v_loss, p_loss = ppo_1.update(memories, envs[0].nb_nodes, configs.graph_pool_type)


        for memory in memories:
            memory.clear_memory()

        mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)
        mean_ms_all_env=sum(ep_ms) / len(ep_rewards)
        mean_de_all_env=sum(ep_de) / len(ep_rewards)

       

        lo.append(v_loss)
        lop.append(p_loss)

        reward_list.append(mean_rewards_all_env)
        ms_list.append(mean_ms_all_env)
        de_list.append(mean_de_all_env)
        sum_list.append(0.25*mean_ms_all_env+0.75*mean_de_all_env)
        
        
       

        log.append([i_update, mean_rewards_all_env])
        print('Episode {}\t Last reward: {:.2f}\t Mean_Vloss: {:.8f}\t '.format(i_update + 1, mean_rewards_all_env, v_loss))
        
        if i_update%100==0:
            file_writing_log=open('./' + 'log_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_log.write(str(log))

            # file_writing_log1=open('./' + 'PLoss_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            # file_writing_log1.write(str(lop))



        
        #Early stopping study
        if (i_update + 1) == window_size:

            torch.save(ppo_1.policy.state_dict(), './{}.pth'.format(
                    str(configs.n_j) + '_1' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high)))
            

            
           
            R_max= mean(reward_list[-window_size:])
            sum_min=mean(sum_list[-window_size:])
            de_min=mean(de_list[-window_size:])
            ms_min=mean(ms_list[-window_size:])

            moving_avg.append(R_max)
            moving_sum.append(sum_min)
            moving_de.append(de_min)
            moving_ms.append(ms_min)

        elif (i_update + 1) > window_size:
            R_i= mean(reward_list[-window_size:])
            sum_min=mean(sum_list[-window_size:])
            de_min=mean(de_list[-window_size:])
            ms_min=mean(ms_list[-window_size:])

            moving_avg.append(R_i)
            moving_sum.append(sum_min)
            moving_de.append(de_min)
            moving_ms.append(ms_min)
            
           
            
            if R_i> R_max:
               
                R_max=R_i  
          
                torch.save(ppo_1.policy.state_dict(), './{}.pth'.format(
                    str(configs.n_j) + '_1' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high)))

                file_writing_rew_window=open('./'+'moving_rew_'+ str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                file_writing_rew_window.write(str(moving_avg))
           
                file_writing_obj = open('./' + 'moving_sum_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                file_writing_obj.write(str(moving_sum))
                
                file_writing_vloss=open('./' + 'VLoss_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                file_writing_vloss.write(str(lo))

                file_writing_ploss=open('./' + 'PLoss_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                file_writing_ploss.write(str(lop))
               
                file_writing_obj1 = open(
                './' + 'valiMSP_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                file_writing_obj1.write(str(moving_ms))
            
                file_writing_obj2 = open(
                './' + 'valiDEVP_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                file_writing_obj2.write(str(moving_de)) 




            else:
                if 100*abs((R_max - R_i)/R_max)>=delta:
                # if 100*((R_max - R_i)/R_max)>=delta:
                   
                    STOP=True
                    file_writing_rew_window=open('./'+'moving_rew_'+ str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_rew_window.write(str(moving_avg))

                    file_writing_obj = open('./' + 'moving_sum_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_obj.write(str(moving_sum))

                    file_writing_vloss=open('./' + 'VLoss_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_vloss.write(str(lo))

                    
                    file_writing_ploss=open('./' + 'PLoss_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_ploss.write(str(lop))

                    file_writing_obj1 = open(
                    './' + 'valiMSP_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_obj1.write(str(moving_ms))       
                
                    file_writing_obj2 = open(
                    './' + 'valiDEVP_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_obj2.write(str(moving_de))

                else:
                    file_writing_rew_window=open('./'+'moving_rew_'+ str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_rew_window.write(str(moving_avg))

                    file_writing_obj = open('./' + 'moving_sum_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_obj.write(str(moving_sum))

                    file_writing_vloss=open('./' + 'VLoss_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_vloss.write(str(lo))

                    
                    file_writing_ploss=open('./' + 'PLoss_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_ploss.write(str(lop))

                    file_writing_obj1 = open(
                    './' + 'valiMSP_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_obj1.write(str(moving_ms))       
                
                    file_writing_obj2 = open(
                    './' + 'valiDEVP_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_obj2.write(str(moving_de))
        
        i_update+=1


if __name__ == '__main__':
    total1 = time.time()
    main()
    total2 = time.time()
    print("Training time is:",total2 - total1)
