import gym
import numpy as np
from gym.utils import EzPickle
from uniform_instance_gen_re import override
from updateEntTimeLB_re import calEndTimeLB
from Params_re import configs
from permissibleLS_re import permissibleLeftShift
from updateAdjMat_re import getActionNbghs
from uniform_instance_gen_re import uni_instance_gen
import math as m
import random
#from math import abs
#Adjacency matrix transfers information of precedence between operations of the same job and meant to be processed on the same machine

class SJSSP(gym.Env, EzPickle):
    def __init__(self,
                 n_j,
                 n_m):
        EzPickle.__init__(self)

        self.step_count = 0
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_tasks = self.number_of_jobs * self.number_of_machines
        #the task id for first column
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        #the task id for last column
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs
        #self.n_nodes=self.number_of_tasks-2

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False 
    
    @override
    def step(self,hour,n_jobs, action_idk, action_idv,eliminate_op,n_add):
    #def step(self, action,eliminate_op):
        #action is a int 0 - 224 for 15x15 for example
        #redundant action makes no effect
        # print("self.partial_sol_sequeence:",self.partial_sol_sequeence)
        # print("action_idk, action_idv:", action_idk, action_idv)
        if action_idv not in self.partial_sol_sequeence:
          
            # UPDATE BASIC INFO:
            row = action_idv // self.number_of_machines
            col = action_idv % self.number_of_machines
            #print("row,column:", row,col)
            self.step_count += 1
            self.finished_mark[row, col] = 1
            dur_a = self.dur[row, col]
            #print("dur_a:", dur_a)
            self.partial_sol_sequeence.append(action_idv)
            
            # UPDATE STATE:
            # permissible left shift
            startTime_a, flag = permissibleLeftShift(hour,n_jobs,a=action_idv, durMat=self.dur, mchMat=self.m, mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs)
            self.flags.append(flag)
            
            self.temp1[row, col] = startTime_a + dur_a
            #print("self.temp1:", self.temp1)
            self.LBs = calEndTimeLB(self.temp1, self.dur_cp)
            
            # update omega or mask
            if action_idv not in self.last_col:
                self.omega[action_idv// self.number_of_machines] += 1
            else:
                
                self.mask[action_idv // self.number_of_machines] = 1
                if row in range(n_jobs):
                    self.dev[row, col]=max(0,((self.temp1[row, col]-self.dev[row, col])))
                    # if self.temp1[row, col]-self.dev[row, col]>=0:
                    #     self.dev[row, col]=((self.temp1[row, col]-self.dev[row, col]))
                    # else:
                    #     self.dev[row, col]=0        
                        
            key_listed=list(self.idd.keys())
            # print("key_listed at step:", key_listed)
            value_listed=list(self.idd.values())
            # print("value list at step:", value_listed)
            candidate_change=[]
            # print("self.omega at step is:", self.omega)
            for j in range(len(self.omega)):
            
                # if self.omega[j] not in self.partial_sol_sequeence:
                
                # position1=value_listed.index(self.omega[j])
                # candidate_change.append(key_listed[position1])
                
            # print("j is :", j)
                # print("for j in range(len(self.omega)):j", j)
               
                if self.omega[j] not in value_listed:
                    # candidate_change.append(self.omega[j])
                    # print("self.omega[j] in step:",self.omega[j] )
                    candidate_change.append(key_listed[value_listed.index(random.choice(value_listed))])
                else: 
                    position1=value_listed.index(self.omega[j])
                    # print("position1=value_listed.index(self.omega[j])", position1)
                    candidate_change.append(key_listed[position1])
                # print("candidate_change at step:", candidate_change)
      

            self.omega_ch=np.asarray(candidate_change)
            self.omega_ch=self.omega_ch.astype(np.int64) 

            # adj matrix
            precd, succd = self.getNghbs(action_idv, self.opIDsOnMchs)
            # print("action_idv,action precd,succd are: ",action_idv,precd, succd)
            self.adj[action_idk] = 0
            self.adj[action_idk, action_idk] = 1
            
            if action_idv not in self.first_col and action_idk!=0 :#translates precedence constraint of operations of the the same job 
                self.adj[action_idk, action_idk - 1] = 1            
            
            if (precd in value_listed) and (succd in value_listed):

                pos_prec=value_listed.index(precd)
                pos_succ=value_listed.index(succd)
                precd_idk=key_listed[pos_prec]
                succd_idk=key_listed[pos_succ]
                # print("action_idk,action_idv,pos_prec,pos_succ,precd_idk,succd_idk:",action_idk,action_idv,pos_prec,pos_succ,precd_idk,succd_idk)
                self.adj[action_idk, precd_idk] = 1
                self.adj[succd_idk, action_idk] = 1 

            if flag and precd != action_idv and succd != action_idv and (precd in value_listed) and (succd in value_listed): # Remove the old arc when a new operation inserts between two operations
                self.adj[succd_idk, precd_idk] = 0
            # print("self.adj first at step prepared:")
            # print(self.adj)
        # prepare for return
        #the change in fea 
        
        fea = np.concatenate((self.LBs.reshape(-1, 1)/configs.et_normalize_coef,
                              self.finished_mark.reshape(-1, 1),self.dev.reshape(-1, 1)/configs.et_normalize_coef), axis=1)
        #print("fea is like", fea)
        if eliminate_op!=[]:
                #print("element is:", e)
            fea=np.delete(fea,eliminate_op,axis=0)
               
        s1=self.dev.sum(axis=0)
        sum1=s1[-1]
        # sum1=np.max(self.dev)

        # print("sum1:", sum1)
        # print("T_max is:", sum1)
        reward = -(0.50*sum1+0.50*self.LBs.max()) + self.max_endTime
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
        self.max_endTime = 0.50*self.LBs.max()+0.50*sum1
        #print("dictionary  is:" , self.idd)
        
        return self.adj, self.dev,fea, reward, self.done(), self.omega_ch, self.mask
    @override
    def reset_add_eliminate(self,data,startTimeOnMchs,opIdsOnmchs,n_add,data_add,eliminate_op,ref_flag,final,final_dev):
        #eliminate op is a list containing operations to be eliminated
        #ref_flag indicates whether we take a ref_schedule at the training(one phase-training whenTRUE , otherwise double_phase_training)
        #final== temp1: les temps de fin d'éxécution des tâches
        
      
        self.m = data[-1]
        # print("self.m is:", self.m)
        
        self.dur = data[0].astype(np.single)
        
        self.dur_cp = np.copy(self.dur)
        
        self.number_of_jobs=self.dur.shape[0]
   
        self.number_of_jobs += n_add
       
        self.step_count = 0
        
        ######si le job est fini normalement on met à jour le nbre de job?
        self.number_of_tasks = self.number_of_jobs * self.number_of_machines
        self.nb_nodes=self.number_of_tasks - len(eliminate_op)
       
        # the task id for first column
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        # the task id for last column
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs
        #ADD jobs with their information
        for i in range(len(data_add[0])):
          self.m =np.vstack((self.m,data_add[1][i]))
          self.dur = np.vstack((self.dur,data_add[0][i]))
        #print("self.dur  shape after add:", self.dur.shape)
        self.dur_cp = np.copy(self.dur)
        self.mchsStartTimes = -configs.high * np.ones_like(self.dur.transpose(), dtype=np.int32)
        # Ops ID on machines
        self.opIDsOnMchs = -self.number_of_jobs * np.ones_like(self.dur.transpose(), dtype=np.int32)
        self.temp1=np.zeros_like(self.dur, dtype=np.single)
        # record action history
        self.partial_sol_sequeence = eliminate_op
        
        #######################we should add flags of executed actions
        self.flags = []
        for i in range(len(eliminate_op)):
            
            self.flags.append(True)
            
        self.posRewards = 0

        # initialize adj matrix: ADD
        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)

        # print("conj_nei_up_stream first:")
        # print(conj_nei_up_stream)

        # first column does not have upper stream conj_nei
        conj_nei_up_stream[self.first_col] = 0
        # last column does not have lower stream conj_nei
        conj_nei_low_stream[self.last_col] = 0

        # print("conj_nei_up_stream after:")
        # print(conj_nei_up_stream)

        self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
        self.adj = self_as_nei + conj_nei_up_stream
        # print("self.adj at rest:")
        # print(self.adj)
        # initialize features:ADD
        self.LBs = np.cumsum(self.dur, axis=1, dtype=np.single)

       
        
        self.finished_mark = np.zeros_like(self.m, dtype=np.single)        
        self.dev=np.zeros_like(self.dur, dtype=np.single)
        # self.dev=np.zeros_like(final, dtype=np.single)
        
        # "old" jobs deviation is initialized to their end times at the reference schedule:
        for k in range (len(final_dev)):   
            self.dev[k, self.number_of_machines-1]=final_dev[k, self.number_of_machines-1]
 
    
        remaining_jobs=[i for i in range(self.number_of_tasks)]
        
        
        job_id=[]
        for e in eliminate_op: 

              remaining_jobs.remove(e)
              
              #update finish_mark
              row_e = e // self.number_of_machines
              col_e = e % self.number_of_machines            
              
              self.temp1[row_e, col_e]=final[row_e, col_e]

              mch_e=int(np.take(self.m,e)-1)
            #   print("mch_e:", mch_e)
              ind1=np.where(opIdsOnmchs[int(mch_e)]==e)[0][0]
            #   print("opIdsOnmchs:", opIdsOnmchs)
            #   print('ind1=np.where(opIdsOnmchs[int(mch_e)]==e)[0][0]:', ind1,e)
         
              self.opIDsOnMchs[mch_e,ind1]=opIdsOnmchs[mch_e,ind1]
              self.mchsStartTimes[mch_e,ind1]=startTimeOnMchs[mch_e,ind1]
              
              if e in self.last_col:
                if ref_flag==True:
                    self.dev[row_e][col_e]= 0
             
                job_id.append( e// self.number_of_machines)  
              self.LBs[row_e, col_e]=self.temp1[row_e, col_e]

        
        self.initQuality = self.LBs.max() if not configs.init_quality_flag else 0
        # self.initQuality = 0 
        self.max_endTime = self.initQuality
        
        #initializin fea tensorconsidering eliminated elements in fea tensor
        fea = np.concatenate((self.LBs.reshape(-1, 1)/configs.et_normalize_coef,
                              # self.dur.reshape(-1, 1)/configs.high,
                              # wkr.reshape(-1, 1)/configs.wkr_normalize_coef,
                              self.finished_mark.reshape(-1, 1),self.dev.reshape(-1, 1)), axis=1)

        
        # Delete from ADJ,fea:
        if eliminate_op!=[]:
            
            self.adj=np.delete(self.adj,eliminate_op,axis=0)
            self.adj=np.delete(self.adj,eliminate_op,axis=1)
            fea=np.delete(fea,eliminate_op ,axis=0)  
        # print("self.adj:",len(self.adj))
        # print("self.adj after elimination at reset:")
        # print(self.adj)
        
        #create the dictionary link
        zip_iterator=zip([i for i in range(len(remaining_jobs))],remaining_jobs)
        # print("zip_iterator:", zip_iterator)
        self.idd=dict(zip_iterator)
        # print("self.idd:", self.idd)
        key_listed=list(self.idd.keys())
        value_listed=list(self.idd.values())
        # print("key_listed:",key_listed)
        # print("value_listed in reset:", value_listed)
        #initialize feasible self.omega & self.omega_ch
        self.omega = self.first_col.astype(np.int64)
        
        for i in range(len(self.omega)):
            while( (self.omega[i] in eliminate_op) and (self.omega[i]not in self.last_col)):
                self.omega[i]+=1
               
        candidate_change=[]
        # print("self.omega:",self.omega)
        for j in range(len(self.omega)):
            # print("j is :", j)
            # print("for j in range(len(self.omega)):j", j)
            # print("value_listed:", value_listed)
            
            if self.omega[j] not in value_listed:
                # candidate_change.append(self.omega[j])
                # print("self.omega[j] in reset:",self.omega[j] )    
                candidate_change.append(key_listed[value_listed.index(random.choice(value_listed))])
            else: 
                position1=value_listed.index(self.omega[j])
                # print("position1=value_listed.index(self.omega[j])", position1)
                candidate_change.append(key_listed[position1])
            # print("candidate_change at reset:", candidate_change)
        
                    
        self.omega_ch=np.asarray(candidate_change)
        self.omega_ch=self.omega_ch.astype(np.int64)
        # print("self.omega_ch:", self.omega_ch)
       
        # initialize mask
    
        self.mask = np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool)
        for i in job_id:
            
            self.mask[i]=1
        

    
        return self.adj, self.dev,fea, self.omega,self.omega_ch, self.mask, self.nb_nodes, self.temp1, self.opIDsOnMchs,self.mchsStartTimes
    










