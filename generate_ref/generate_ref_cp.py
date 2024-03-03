import numpy as np
from uniform_instance_gen_re import uni_instance_gen
import pickle


from docplex.cp.config import context
# context.solver.local.execfile = '/home/nehammami/cpoptimizer/bin/x86-64_linux/cpoptimizer'
from docplex.cp.model import *
context.solver.log_output = None

from obj_ref import *

def optimize(instance, dur, seq,n_mch, n_job,max_time = 1000000, time_limit = 100, threads = 2):
    model = CpoModel('SimpleJobShop')
    interval_vars = dict()
    
    # Create variables
    for task in instance.tasks:
        interval_vars[task] = interval_var(start = (0, max_time), end = (0, max_time), size = task.length, name = 'interval'+str(task.name))
    
    # Precedence and blocking constraints
    for task in instance.tasks:
        if task.next_task:
            model.add(start_of(interval_vars[task.next_task]) >= end_of(interval_vars[task]))
            
    # No overlap constraints
    for machine in instance.machines:
        machine_sequence = sequence_var([interval_vars[task] for task in machine.tasks])
        model.add(no_overlap(machine_sequence))
            
    # Minimize the makespan
    obj_var = integer_var(0, max_time, 'makespan')
    for task in instance.tasks:
        model.add(obj_var >= end_of(interval_vars[task]))
    model.minimize(obj_var)
    
    # Define solver and solve
    sol = model.solve(TimeLimit= time_limit, Workers = threads)
    
    solution = Solution(instance)
    
    # Print out solution and return it
    for job in instance.jobs:
        for task in job.tasks:
            # print(task.name)
            start = sol.get_value(interval_vars[task])[0]
            end = sol.get_value(interval_vars[task])[1]
            # print('Start: %f'%start)
            # print('End: %f'%end)
            solution.add(task, start, end, task.machine)
    tasks_sorted = sorted(solution.tasks, key=lambda task: sol.get_value(interval_vars[task])[0])
    mchsStartTimes = -1 * np.ones_like(dur.transpose(), dtype=np.int32)
    opIDsOnmchs=-1 * np.ones_like(dur.transpose(), dtype=np.int32)
    end=np.ones_like(dur, dtype=np.int32)
    solution_sequence=[]
    for task in tasks_sorted:
                    # print("sol seq per step:", solution_sequence)
                    solution_sequence.append((task.job.id-1)*n_mch+(task.order-1))
                    
                    # print(task.name,sol.get_value(interval_vars[task])[0],sol.get_value(interval_vars[task])[1], task.job.id-1, task.machine.id )

   
    l=[i for i in range(len(mchsStartTimes))]
    l1= [i for i in range(len(dur))]
    mch_start_dic={k: [] for k in l}
    op_id_dic={k: [] for k in l}
    end_dic={k: [] for k in l1}
    # print("mch_start_dic:", mch_start_dic,l)
    for key in mch_start_dic:
        for task in tasks_sorted:
            if task.machine.id==key:
                mch_start_dic[key].append(sol.get_value(interval_vars[task])[0])
                op_id_dic[key].append((task.job.id-1)*n_mch+(task.order-1))
      

    for i in range(len(mchsStartTimes)):
            for j in range(len(mchsStartTimes[0])):
            
                        opIDsOnmchs[i][j]=op_id_dic[i][j]
                        mchsStartTimes[i][j]=mch_start_dic[i][j]
    # print(mchsStartTimes, opIDsOnmchs)

    # PREPARE END DIC 
    for key in end_dic: 
        for task in tasks_sorted:
            if task.job.id-1==key:
                 end_dic[key].append(sol.get_value(interval_vars[task])[1])
                 
    # print("end_dic is", end_dic)
    for i in range(len(end)):
        for j in range(len(end[0])):
            end[i][j]=end_dic[i][j]

            
        
    
    return solution,mchsStartTimes, opIDsOnmchs,end, solution_sequence

def optimize_and_visualize(n_mch, n_job,dur,seq, time_limit = 100, threads = 2):
    reader = reading(dur,seq)
    instance = reader.get_instance()
    solution,mchsStartTimes, opIDsOnmchs,end, solution_sequence = optimize(instance,dur, seq,n_mch, n_job, time_limit = time_limit, threads = threads)
    # print("sol visualize:", solution_sequence)
    # print(seq)
    return dur, seq, mchsStartTimes, opIDsOnmchs,end, solution_sequence
    # solution.visualize(time_factor = 1, time_grid = 50)

    
if __name__ == '__main__':
    j = 15
    m = 15
    l = 1
    h = 99
    seed = 200

    np.random.seed(seed)

    #data = np.array([uni_instance_gen(n_j=j, n_m=m, low=l, high=h)])
    #print("data is:", data)
    
    dur,seq=uni_instance_gen(n_j=j, n_m=m, low=l, high=h)

    mchsStartTimes_dic, opIDsOnmchs_dic,end_dic, solution_sequence_dic,dur_dic,seq_dic={},{},{},{},{},{}
    dur,seq,mchsStartTimes, opIDsOnmchs,end, solution_sequence=optimize_and_visualize(m,j,dur,seq, 0.02,1)
    # print("solution_main", solution_sequence)
    # print("end.shape, mchstarts.shape", end.shape, mchsStartTimes.shape)
    dur_dic[0]=dur
    seq_dic[0]=seq
    mchsStartTimes_dic[0]=mchsStartTimes
    opIDsOnmchs_dic[0]=opIDsOnmchs
    end_dic[0]= end
    solution_sequence_dic[0]=solution_sequence
    # print("mchsStartTimes:", mchsStartTimes)
    # print("opIDsOnmchs:", opIDsOnmchs)
    mchs_file=open("mchs_cp.pkl", "wb")
    pickle.dump(mchsStartTimes_dic, mchs_file)
    mchs_file.close()

 
    opids_file=open("opids_cp.pkl", "wb")
    pickle.dump(opIDsOnmchs_dic, opids_file)
    opids_file.close()
 
    end_file=open("end_cp.pkl", "wb")
    pickle.dump(end_dic, end_file)
    end_file.close()

    sol_file=open("sol_cp.pkl", "wb")
    pickle.dump(solution_sequence_dic, sol_file)
    sol_file.close()

    dur_file=open("dur_cp.pkl", "wb")
    pickle.dump(dur_dic, dur_file)
    dur_file.close()

    seq_file=open("seq_cp.pkl", "wb")
    pickle.dump(seq_dic, seq_file)
    seq_file.close()

    # print("data is:", dur,seq)
    # print("solution", solution_sequence_dic)
    # print("mch,opid,end:", mchsStartTimes, opIDsOnmchs, end)


