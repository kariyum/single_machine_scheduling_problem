import random
import numpy as np
import matplotlib.pyplot as plt

class Job:
    def __init__(self, t) -> None:
        self.r, self.p, self.d, self.id = t
        self.remaining_time = self.p

    def __repr__(self) -> str:
        return str("(" + str(self.r) +", "+ str(self.p) +", "+ str(self.d) +", "+ str(self.id)+")")


def generate_data(n):
    """
        n number of jobs
        generates random release release dates (rj) using the poisson process
        process time (pj) is a random integer generated randomly ranging from 0 to 10
        due dates (dj) follow this formula: rj + pj + random integer between 0 and 10
        jobs is a list of tuples (rj, pj, dj, id) 
    """
    release_date = np.random.poisson(lam=12, size=n)
    process_time = [random.randint(1, 10) for _ in range(len(release_date))]
    due_date = [x[0] + x[1] + random.randint(0, 10) for x in zip(release_date, process_time)]
    jobs = list(zip(release_date, process_time, due_date, range(n)))
    return jobs

def drawGantt(res):
    """
        utility function that draw GANTT diagram of the solution
        takes solution as an argument and draws the corresponding GANTT diagram
    """
    aux_res = res.copy()
    print(res)
    for i in range(1, len(res)):
        res[i] = (max(res[i-1][0] + res[i-1][1], res[i][0]), res[i][1], res[i][2], res[i][3])
    
    # plotting the GANTT diagram
    plt.figure(figsize = (10, 10))
    plt.subplot(2, 1, 1)
    plt.barh(y = [str(x[3]) for x in res], width = [x[1] for x in res], left= [x[0] for x in res])
    plt.title("Solution plot GANTT")
    plt.xlim([min([x[0] for x in aux_res]), max([max(res[i-1][0] + res[i-1][1], res[i][0]) + x[1] for x in aux_res])])
    plt.subplot(2, 1, 2)
    plt.barh(y = [str(x[3]) for x in aux_res], width = [x[1] for x in aux_res], left= [x[0] for x in aux_res])
    plt.xlim([min([x[0] for x in aux_res]), max([max(res[i-1][0] + res[i-1][1], res[i][0]) + x[1] for x in aux_res])])
    plt.title("Data plot GANTT")
    plt.show()

def schedule(data):
    """
        utility function that solves the problem through EDD
        data contains : release date, due date, process time, jobID
        data = [(r0, p0, d0, jobID), ..., (rj, pj, dj, jobID)...]
        sort by release date then sort by due date
    """
    sorted_jobs = sorted(data, key = lambda job : (job[0], job[1]))
    return sorted_jobs

def getScore(sol):
    """
     only works for completed solution without interruption
     helper function that scores the solution (input)
     input: a solution (rj, pj, dj, id)
     output: a score if < means some jobs couldn't meet the deadline else we don't know.
    """
    s = sol.copy()
    score = s[0][2] - (s[0][0] + s[0][1])
    for i in range(1, len(s)):
        s[i] = (max(s[i-1][0] + s[i-1][1], s[i][0]), s[i][1], s[i][2], s[i][3])
        score += s[i][2] - (s[i][0] + s[i][1])
    return score

def heurisitc(sol, data, objective_function):
    """
     input: a semi completed solution (sol, data: list of Job class type)
     completes the solution with the relaxed constraint which is preemption
     output: score
    """
    remaining_jobs_length = len(data) - len(sol)
    remaining_jobs = list(filter(lambda x : x.id not in [y.id for y in sol], data))
    
    if (len(remaining_jobs) != remaining_jobs_length):
        print("Remaining jobs error. Check heuristic function.")
        exit(-1)
    # execute tasks with priority to due dates 
    print("WHILE") 
    while (len(remaining_jobs) != 0):
        print("remaining_jobs = ", remaining_jobs)
        stops = list(set([job.r for job in remaining_jobs]))
        stops = sorted(stops)
        if (len(stops) == 0):
            break
        for i, stop in enumerate(stops):
            # stop == rj where new jobs are avaiable, choose one of them with priority to lower due date
            availble_jobs = list(filter(lambda x : x.r == stop, remaining_jobs))
            availble_jobs = sorted(availble_jobs, key= lambda x : (x.r, x.d))
            
            # execute the job at 0 index having lowest release date and due date
            next_r = stops[i+1] if (i < len(stops)-1) else 1e10
            how_long = next_r - stop + 1
            availble_jobs[0].remaining_time -= min(availble_jobs[0].remaining_time, how_long)
            sol.append(Job((availble_jobs[0].r, availble_jobs[0].p, availble_jobs[0].d, availble_jobs[0].id)))
        remaining_jobs = list(filter(lambda x : x.remaining_time != 0, remaining_jobs))
    print(sorted(sol, key= lambda x : x.id))


def branchAndBound(data, heuristic_function, objective_function):
    """
        data contains : release date, due date, process time, jobID
        lower bound is calculated with the relaxation of one constraint which is the possibility to interrupt 
        a job and execute another one
    """
    # at first the solution is empty
    # 1. insert one of the remaining jobs
    # 2. calculate the lower bound for each inserted job
    # 3. branch & bound
    jobs = [Job(t) for t in data]
    global_res = list()
    def rec(sol, data, lower_bound):
        # print(sol)
        if (len(sol) == len(data)):
            print(sol)
            return sol, lower_bound
        # if (heuristic(sol) > lower_bound):
        #     # bound
        #     print("bound")
        #     return []
        # branch
        lower_bound = min(lower_bound, heuristic_function(sol))
        # remaining jobs start from data[len(jobs):-1]
        # call this function for each remaining job
        for job in data[len(sol):]:
            local_sol = sol.copy()
            local_sol.append(job)
            # print("local_sol", local_sol)
            local_res, lb = rec(local_sol, data, lower_bound)
            # print("here", job)
            lower_bound = min(lower_bound, lb)
        return sol, lower_bound
    rec(list(), data, 1e10)
    return global_res


if __name__ == '__main__' :
    data = generate_data(12)
    # res = schedule(data)
    # drawGantt(res.copy())
    # print(getScore(res))
    # print(res)
    
    # res = branchAndBound(data)
    # print(res)
    # drawGantt(res)
    data = [Job(t) for t in data]
    heurisitc(data[:3], data, getScore)
