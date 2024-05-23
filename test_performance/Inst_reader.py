import json
import os
import pickle
#import math

class Instance:
    def __init__(self, name, jobs, machines):
        self.name = name
        self.machines = machines
        self.jobs = jobs
        self.tasks = []
        for job in self.jobs:
            for task in job.tasks:
                self.tasks.append(task)

class Machine:
    def __init__(self, id):
        self.id = id
        self.tasks = set()
    
    def add_task(self, task):
        self.tasks.add(task)

class Job:
    def __init__(self, id):
        self.id = id
        self.tasks = [] # tasks are in a given order
        self.prev_end=[] #for the definition of eliminated tasks at the arrival of jobs
        self.ref=0  #for the calculus of deviation 
        
    def append_task(self, task):
        if len(self.tasks) > 0:
            self.tasks[-1].next_task = task
            task.prev_task = self.tasks[-1]
        self.tasks.append(task)
        task.job = self

class Task:
    def __init__(self, name, order, machine, length):
        self.name = name
        self.order=order
        self.machine = machine
        self.machine.add_task(self)
        self.length = length
        self.job = None
        self.next_task = None
        self.prev_task = None
    
    def __str__(self):
        return self.name
    
class reading:
    # The work is done per hour
    # read needed dictionaries: duration, end times,machines sequence 
    # ==> extract number of jobs, number of machines
    # 1- Read needed dictionaries: duration, end times,machines sequence 
    def __init__(self, hour,prev, dur, seq, start, end, eliminated):
             
    #  2- Prepare jobs, tasks and machines
        jobs = []
        machines = []
        # task_counter = 0
        num_jobs = len(dur[hour])
        num_machines = len(dur[hour][0])
        # we introduce number of tasks to treat the general problem of JSSP where the number of tasks is not necessarily the same as the number of machines (per job)
        num_tasks = len(dur[hour][0]) 
        machines = [Machine(id = i) for i in range(num_machines)]
        job_counter = 1
        for i in range(num_jobs):
            job = Job(job_counter)
            jobs.append(job)
           
            
            for j in range(num_tasks):
                name = 'j_%i_t_%i' % (job_counter, j+1)
                task_machine_id = seq[hour][job_counter-1][j]
                task_machine = machines[task_machine_id-1]
                task_length = int(dur[hour][job_counter-1][j])
                task = Task(name, j+1,task_machine, task_length)
                job.append_task(task) 

            if hour!=0 and eliminated[hour]!=[]:
                if job_counter <= len(end[prev]):
                    job.prev_end.append(end[prev][job_counter-1][-1])
                    
                if job_counter <= len(dur[0]):
                    # Add end of the reference for the calculation of the deviation
                    job.ref=end[0][job_counter-1][-1]

            job_counter += 1
        self.instance = Instance(hour, jobs, machines)

    def get_instance(self):
        return self.instance

        
class Solution:
    def __init__(self, instance):
        self.instance = instance
        self.solution = dict()
        self.tasks = list()
        
    def add(self, task, start, end):
        self.solution[task] = (start, end)
        self.tasks.append(task)
        
    def get_start_time(self, task):
        return self.solution[task][0]
        
    def get_end_time(self, task):
        return self.solution[task][1]
        
    def get_makespan(self):
        return max([self.get_end_time(task) for task in self.tasks])
        
    # def visualize(self, path = './writing.txt', line_spacing = 10, line_height = 30, time_factor = 1, time_grid = 10):
    #     instance = self.instance
    #     solution = self
    #     color_data = json.load(open('colors.json','r'))
    #     f = open(path, 'w')
    #     # first draw the machines
    #     for machine in instance.machines:
    #         f.write('<div style="position:absolute; left: 20px; top: %ipx; height: %ipx; width: %ipx; border-style: solid; border-width: 1px; border-color: black; text-align: center">M%i</div>\n'%(20+(line_spacing+line_height)*machine.id, line_height, 30, machine.id))
    #     x_offset = 50
    #     # draw the time grid
    #     max_time = self.get_makespan()
    #     max_height = len(instance.machines)*(line_spacing + line_height)+line_height
    #     for t in range(time_grid, int(max_time+time_grid*2), time_grid):
    #         f.write('<div style="position:absolute; left: %ipx; top: 0px; height: %ipx; width: 0px; border-style: solid; border-width: 0.5px; border-color: black"></div>\n'%(x_offset + t*time_factor, max_height))
    #         f.write('<div style="position:absolute; left: %ipx; top: %ipx; height: 20px; width: 50px; margin-left: -25px; border: 0; text-align: center">%i</div>\n'%(x_offset + t*time_factor, max_height+line_spacing, t))
    #     # then draw each task
    #     for task in instance.tasks:
    #         start = solution.get_start_time(task)
    #         end = solution.get_end_time(task)
    #         duration = end-start
    #         bg_color = color_data[str(len(instance.jobs))][task.job.id-1][3]
    #         text_color = color_data[str(len(instance.jobs))][task.job.id-1][4]
    #         f.write('<div style="position:absolute; left: %ipx; top: %ipx; height: %ipx; width: %ipx; border-style: solid; border-width: 1px; border-color: black; text-align: left; font-size: 8px; background-color: %s; color: %s"><div style="position: absolute; text-align: center; width: %ipx; margin: 0; top: 50%%; -ms-transform: translateY(-50%%); transform: translateY(-50%%)">%s</div></div>\n'%(x_offset + start*time_factor, 20+(line_spacing+line_height)*task.machine.id, line_height, task.length*time_factor, bg_color, text_color, task.length*time_factor, task.name))
    #         if duration-task.length > 0:
    #             f.write('<div style="position:absolute; left: %ipx; top: %ipx; height: %ipx; width: %ipx; border-style: solid; border-width: 1px; border-color: black; text-align: left; font-size: 8px; background-color: %s; color: %s; background-image: radial-gradient(black 50%%, transparent 50%%); background-size: 2px 2px"></div>\n'%(x_offset + (start+task.length)*time_factor, 20+(line_spacing+line_height)*task.machine.id+line_height/4, line_height/2, (duration-task.length)*time_factor, bg_color, text_color))
    #     f.write('<div style="position: absolute; left: 20px; top: %ipx"><u>Makespan:</u> %i</div>'%(max_height+50, max_time))
    #     f.close()
