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
    tuples = list(zip(release_date, process_time, due_date, range(n)))
    jobs = [Job(t) for t in tuples]
    return jobs

def drawGantt(res : list[Job], data : list[Job], block= True):
    """
        utility function that draws GANTT diagram of the solution
        takes a solution as an argument and draws the corresponding GANTT diagram
    """
    
    # plotting the GANTT diagram
    plt.figure(figsize = (10, 10))
    
    plt.subplot(2, 1, 1)
    plt.barh(y = [str(job.id) for job in res], width = [job.p for job in res], left= [job.r for job in res])
    plt.title("Solution plot GANTT")
    # plt.xticks(np.arange(min([job.r for job in res]), max([job.r + job.p for job in res]) + 1, 1))
    plt.xticks(list(set([job.r for job in res] + [job.r + job.p for job in res])))

    plt.subplot(2, 1, 2)
    plt.barh(y = [str(job.id) for job in data], width = [job.p for job in data], left= [job.r for job in data])
    # plt.xlim(left= min([job.r for job in data]), right= max([job.r + job.p for job in res]) + 1)
    plt.xticks(list(set([job.r for job in data] + [job.r + job.p for job in data] + [max([job.r + job.p for job in res]) + 1])))
    plt.title("Data plot GANTT")
    plt.show(block= block)

def schedule(data : list[Job]):
    """
        utility function that solves the problem through EDD
        data contains : release date, due date, process time, jobID
        data = [(r0, p0, d0, jobID), ..., (rj, pj, dj, jobID)...]
        sort by release date then sort by due date
    """
    sorted_jobs = sorted(data, key = lambda job : (job.r, job.d))
    sorted_jobs = [Job((job.r, job.p, job.d, job.id)) for job in sorted_jobs]
    for i in range(1, len(sorted_jobs)):
        sorted_jobs[i].r = max(sorted_jobs[i-1].r + sorted_jobs[i-1].p, sorted_jobs[i].r)
    return sorted_jobs

def getScore(sol : list[Job]):
    """
     only works for completed solution without interruption
     helper function that scores the solution (input)
     input: a solution (rj, pj, dj, id)
     output: a score if < means some jobs couldn't meet the deadline else we don't know.
    """
    score = 0
    for job in sol:
        # score += job.d - (job.r + job.p)
        score = min(score, job.d - (job.r + job.p)) # Lmax = retard maximal
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
        raise ValueError("Remaining jobs error. Check heuristic function.")

    # execute tasks with priority to due dates 
    if (len(remaining_jobs) != 0):
        # print("remaining_jobs = ", remaining_jobs)
        stops = list(set([job.r for job in remaining_jobs]))
        stops = sorted(stops)
        while (len(stops) != 0 and len(remaining_jobs) != 0) :
            i = 0
            stop = stops[0]
            next_stop = stops[i+1] if (i < len(stops)-1) else 1e10
            stops.pop(0)
            # stop >= rj where new jobs are avaiable
            available_jobs = list(filter(lambda x : x.r <= stop, remaining_jobs))
            available_jobs = sorted(available_jobs, key= lambda x : x.p / x.d)
            # print("Available jobs {}, Stop {}, stops {}".format(available_jobs, stop, stops))
            if (len(available_jobs) == 0):
                print("Machine is waiting")
                continue
            job_to_execute = available_jobs[0]
            available_jobs.pop(0)
            remaining_jobs = list(filter(lambda x : x.id != job_to_execute.id, remaining_jobs))

            # execute the job at 0 index having lowest release date and due date
            how_long = next_stop - stop
            executed_for = job_to_execute.remaining_time
            job_to_execute.remaining_time -= min(job_to_execute.remaining_time, how_long)
            executed_for -= job_to_execute.remaining_time
            sol.append(Job((job_to_execute.r, executed_for, job_to_execute.d, job_to_execute.id)))
            if (job_to_execute.remaining_time != 0):
                remaining_jobs.append(Job((job_to_execute.r, job_to_execute.remaining_time, job_to_execute.d, job_to_execute.id)))
            if (executed_for + stop < next_stop):
                stops.append(executed_for + stop)
            stops = sorted(list(set(stops)))
        # remaining_jobs = list(filter(lambda x : x.remaining_time != 0, remaining_jobs))
    # print(sol)
    # drawGantt([(job.r, job.p, job.d, job.id) for job in sol])
    return sol, objective_function(sol)

def getScoreHeuristic(sol : list[Job]):
    """
     should handle feasable and non feasable solution
     inputs: a solution
     returns a score
     this function should is equivalent to getScore function when all jobs are executed without interruption
     TO VERIFY !
    """
    # sol : [Job()]
    # convert each rj to start execution first
    # we are going to group all exeuction chunks of the same job together
    # compute the difference between completion time and due date
    # add it to the score
    aux = sol.copy()
    # convert each rj to the instant when it started the execution
    for i in range(1, len(sol)):
        aux[i].r = max(sol[i-1].p + sol[i-1].r, sol[i].r)
    
    # group all execution chunks of the same job together
    hash_map = dict()
    for job in aux:
        if (job.id in hash_map.keys()):
            hash_map[job.id].append(job)
        else:
            hash_map[job.id] = [job]
    
    # for each grouped chunks retrieve the last executed chunk
    for id in hash_map.keys():
        hash_map[id] = max(hash_map[id], key= lambda x : x.r)
    
    # now it's just a matter of adding up difference between job.duedate and job.r + job.p
    score = 0
    for id in hash_map.keys():
        # score += hash_map[id].d - (hash_map[id].r + hash_map[id].p)
        score = min(score, hash_map[id].d - (hash_map[id].r + hash_map[id].p))
    # print(hash_map)
    return score

def branchAndBound(data, heuristic_function, objective_function):
    """
        NOT FULLY IMPLEMENTED !
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

def simulate(n= 100):
    h = list()
    while (n != 0):
        n -= 1
        data = generate_data(4)
        # print(data)

        res = schedule(data)
        # drawGantt(res.copy())
        # print(getScore(res))
        # print(res)
            
        # res = branchAndBound(data)
        # print(res)
        # drawGantt(res)
        data = [Job(t) for t in data]
        res2 = heurisitc([], data, getScoreHeuristic)
        # print(res)
        # drawGantt(res)
        if (getScoreHeuristic([Job((x[0], x[1], x[2], x[3])) for x in res]) != getScore(res)):
            print("NO")
            
        h.append((getScoreHeuristic([Job((x[0], x[1], x[2], x[3])) for x in res]), res2))
    # print(h)
    plt.plot(range(len(h)), [x[0] for x in h])
    plt.plot(range(len(h)), [x[1] for x in h])
    plt.show()

if __name__ == '__main__' :
    data = generate_data(14) # make sure that this data is constant
    res = schedule(data)
    print(getScore(res))
    drawGantt(res, data, block= False)
    # # print(res)
    res2, score = heurisitc([], data, getScoreHeuristic)
    print(score)
    drawGantt(res2, data)
    # res2 = heurisitc([], [Job(t) for t in data], getScoreHeuristic)
    # simulate(100)
    # input()