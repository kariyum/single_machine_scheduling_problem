import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

class Job:
    def __init__(self, t) -> None:
        self.r, self.p, self.d, self.id = t
        self.remaining_time = self.p

    def __repr__(self) -> str:
        return str("(" + str(self.r) +", "+ str(self.p) +", "+ str(self.d) +", "+ str(self.id)+")")

lam_global = 12
process_time_range = (1, 10)
due_date_range = (0, 10)
def generate_data(n, p= False):
    """
    :n number of jobs
    :generates random release release dates (rj) using the poisson process
    :process time (pj) is a random integer generated randomly ranging from 0 to 10
    :due dates (dj) follow this formula: rj + pj + random integer between 0 and 10
    :jobs is a list of tuples (rj, pj, dj, id) 
    """
    release_date = np.random.poisson(lam=lam_global, size=n)
    process_time = [random.randint(process_time_range[0], process_time_range[1]) for _ in range(len(release_date))]
    due_date = [x[0] + x[1] + random.randint(due_date_range[0], due_date_range[1]) for x in zip(release_date, process_time)]
    tuples = list(zip(release_date, process_time, due_date, range(n)))
    jobs = [Job(t) for t in tuples]
    if (p):
        for j in jobs:
            print(j)
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
    :utility function that solves the problem through EDD
    :data contains : release date, due date, process time, jobID
    :data = [(r0, p0, d0, jobID), ..., (rj, pj, dj, jobID)...]
    :sort by release date then sort by due date
    """
    sorted_jobs = sorted(data, key = lambda job : (job.r, job.d))
    sorted_jobs = [Job((job.r, job.p, job.d, job.id)) for job in sorted_jobs]
    for i in range(1, len(sorted_jobs)):
        sorted_jobs[i].r = max(sorted_jobs[i-1].r + sorted_jobs[i-1].p, sorted_jobs[i].r)
    return sorted_jobs, getScore(sorted_jobs)

def getScore(sol : list[Job]):
    """
    :only works for completed solution without interruption
    :helper function that scores the solution (input)
    :input: a solution (rj, pj, dj, id)
    :output: a score if < means some jobs couldn't meet the deadline else we don't know.
    """
    score = 0
    for job in sol:
        # score += job.d - (job.r + job.p)
        score = min(score, job.d - (job.r + job.p)) # Lmax = retard maximal
    return score

def heurisitc(sol : list[Job], data : list[Job]):
    """
    :input: a semi completed solution (sol, data: list of Job class type)
    :completes the solution with the relaxed constraint which is preemption
    :output: score
    """
    data = [Job((j.r, j.p, j.d, j.id)) for j in data]
    sol = [Job((s.r, s.p, s.d, s.id)) for s in sol]
    remaining_jobs_length = len(data) - len(sol)
    remaining_jobs = list(filter(lambda x : x.id not in [y.id for y in sol], data))

    if (len(remaining_jobs) != remaining_jobs_length):
        raise ValueError("Remaining jobs error. Check heuristic function. {} {}".format(len(remaining_jobs), remaining_jobs_length))

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
            available_jobs = sorted(available_jobs, key= lambda x : x.d)
            # print("Available jobs {}, Stop {}, stops {}".format(available_jobs, stop, stops))
            if (len(available_jobs) == 0):
                # print("Machine is waiting")
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
    # drawGantt([Job((job.r, job.p, job.d, job.id)) for job in sol], data)
    return sol, getScoreHeuristic([Job((s.r, s.p, s.d, s.id)) for s in sol])

def getScoreHeuristic(sol : list[Job]):
    """
    :should handle feasable and non feasable solution
    :inputs: a solution list[Job]
    :returns a score
    :this function should be equivalent to getScore function when all jobs are executed without interruption
    """
    # sol : [Job()]
    # convert each rj to start execution first
    # we are going to group all exeuction chunks of the same job together
    # compute the difference between completion time and due date
    # add it to the score
    aux = [Job((s.r, s.p, s.d, s.id)) for s in sol]
    
    # convert each rj to the instant when it started the execution
    for i in range(1, len(aux)):
        aux[i].r = max(aux[i-1].p + aux[i-1].r, aux[i].r)
    
    # group all execution chunks of the same job together
    hash_map = dict()
    for job in aux:
        if (job.id in hash_map.keys()):
            hash_map[job.id].append(job)
        else:
            hash_map[job.id] = [job]
    # print(hash_map)
    # print([len(val) > 1 for val in hash_map.values()])
    realisable = all([len(val) == 1 for val in hash_map.values()])
    # print(realisable)
    # for each grouped chunks retrieve the last executed chunk
    # print(hash_map)
    for id in hash_map.keys():
        hash_map[id] = max(hash_map[id], key= lambda x : x.r)
    
    # now it's just a matter of adding up difference between job.duedate and job.r + job.p
    score = 0
    # print(aux)
    for id in hash_map.keys():
        # score += hash_map[id].d - (hash_map[id].r + hash_map[id].p)
        score = min(score, hash_map[id].d - (hash_map[id].r + hash_map[id].p))
    return score, realisable


@dataclass
class NodeState:
    """Class for keeping track of the node state in the queue."""
    remaining_jobs: list[Job]
    curr_sol: list[Job]


def branchAndBound(data : list[Job], v= False):
    """
    :data contains : release date, due date, process time, jobID
    :lower bound is calculated with the relaxation of one constraint which is the possibility to interrupt 
    :a job and execute another one
    """
    # at first the solution is empty
    # 1. insert one of the remaining jobs
    # 2. calculate the lower bound for each inserted job
    # 3. branch & bound
    
    data = [Job((t.r, t.p, t.d, t.id)) for t in data]
    # contains tuple for the parent data (remaining jobs) and current non completed solution
    jobs_queue : list[NodeState] = [NodeState(remaining_jobs= [Job((t.r, t.p, t.d, t.id)) for t in data], curr_sol= [])]
    best_solution, b = schedule([Job((t.r, t.p, t.d, t.id)) for t in data]) # best solution found so far REALISABLE
    while len(jobs_queue) != 0:
        if v == True:
            print("Queued jobs count: {}".format(len(jobs_queue)))
        # print("POP")
        state : NodeState = jobs_queue.pop(0)
        # print("State curr sol, ", state.curr_sol)
        # state.remaining_jobs = [j for j in state.remaining_jobs if j not in state.curr_sol]
        # print(len(state.remaining_jobs))
        for job in state.remaining_jobs:
            # new_remaining_jobs = [j for j in state.remaining_jobs if j.d != job.d]
            remaining_jobs = [Job((j.r, j.p, j.d, j.id)) for j in state.remaining_jobs if j.id != job.id]
            new_solution = [Job((j.r, j.p, j.d, j.id)) for j in state.curr_sol] + [Job((job.r, job.p, job.d, job.id))]
            solution, (h_score, realisable) = heurisitc(sol= [Job((j.r, j.p, j.d, j.id)) for j in new_solution], data= [Job((t.r, t.p, t.d, t.id)) for t in data])
            if h_score <= b: # bound -50 < -30
                continue
            else: # branch
                jobs_queue.append(NodeState(remaining_jobs, [Job((j.r, j.p, j.d, j.id)) for j in new_solution]))
            if realisable and h_score > b:
                # print("solution {} \nrealisable {}".format(solution, realisable))
                # print("h: {} b: {}".format(h_score, b))
                # print("Solution ", solution)
                b = max(b, h_score)
                best_solution = [Job((j.r, j.p, j.d, j.id)) for j in solution]
                # print("Best solution,", best_solution)
    # print("Best solution inside {}".format(best_solution))
    # drawGantt(best_solution, data)
    return getScoreHeuristic(best_solution)[0] #best_solution, getScoreHeuristic(best_solution)

def drawHeuristic(res : list[Job], data : list[Job], block= True):
    # print(res)
    jobs = [Job((j.r, j.p, j.d, j.id)) for j in res]
    res = [(j.r, j.p, j.d, j.id) for j in res]
    for i in range(1, len(res)):
        res[i] = (max(res[i-1][0] + res[i-1][1], res[i][0]), res[i][1], res[i][2], res[i][3])
    
    # plotting the GANTT diagram
    # fig = plt.figure(figsize = (10, 10))
    fig, (ax1,ax2) = plt.subplots(nrows= 2, sharex=True, figsize=(10, 10))
    plt.subplots_adjust(hspace=.0)
    ax1.barh(y = [str(x[3]) for x in res], width = [x[1] for x in res], left= [x[0] for x in res])

    # plt.xlim([min([x[0] for x in aux_res]), max([max(res[i-1][0] + res[i-1][1], res[i][0]) + x[1] for x in aux_res])])
    # plt.xticks(np.arange(min([x[0] for x in aux_res]), max([max(res[i-1][0] + res[i-1][1], res[i][0]) + x[1] for x in aux_res]) + 1, 1))
    plt.xticks(list(set([r[0] for r in res] + [r[0] + r[1] for r in res])))
    # plt.subplot(2, 1, 2, sharex=True)
    ax2.barh(y = [str(j.id) for j in data], width = [j.p for j in data], left= [j.r for j in data])
    # plt.xlim([min([x[0] for x in aux_res]), max([max(res[i-1][0] + res[i-1][1], res[i][0]) + x[1] for x in aux_res])])
    plt.xticks(list(set([j.r for j in jobs] + [j.r + j.p for j in jobs] + [r[0] + r[1] for r in res])))
    # plt.tight_layout()
    plt.show(block= block)

def geneticAlgorithm(data: list[Job], pop_size= 50):
    jobs = [Job((j.r, j.p, j.d, j.id)) for j in data]
    def evaluate(agents: list[Job]):
        return [(agent, getScoreHeuristic(agent)) for agent in agents]
    
    def select(agents: list[(Job, int)]):
        aux = sorted(agents, key= lambda t : t[1], reverse= True)[:int(0.2*pop_size)]
        # for i in range(10):
        #     print(aux[i][1])
        return [a[0] for a in aux], (aux[0][0], aux[0][1][0], aux[0][1][1])

    def crossover(agents: list[Job]):
        offsprings = []

        for _ in range(int((pop_size - len(agents))/2)):
            p1 = random.choice(agents)
            p2 = random.choice(agents)

            offspring = []
            for j1, j2 in zip(p1, p2):
                j1_job_in_offspring = j1.id in [x.id for x in offspring]
                j2_job_in_offspring = j2.id in [x.id for x in offspring]
                if j2_job_in_offspring and j1_job_in_offspring:
                    jobs_in_offspring = [x.id for x in offspring]
                    jobs_not_inoffspring = [job for job in p1 if not job.id in jobs_in_offspring]
                    # print("Jobs not in offspring {} jobs in offspring {}".format(jobs_not_inoffspring, offspring))
                    offspring.append(random.choice(jobs_not_inoffspring))
                    continue
                if not j2_job_in_offspring and not j1_job_in_offspring:
                    if (random.random() < 0.5):
                        offspring.append(Job((j1.r, j1.p, j1.d, j1.id)))
                    else:
                        offspring.append(Job((j2.r, j2.p, j2.d, j2.id)))
                    continue
                if j2_job_in_offspring:
                    offspring.append(Job((j1.r, j1.p, j1.d, j1.id)))
                else:
                    offspring.append(Job((j2.r, j2.p, j2.d, j2.id)))
            offsprings.append(offspring)
        
        return offsprings

    def mutate(agents: list[Job]):
        new_agents = []
        for agent in agents:
            if random.random() < 0.1:
                a = random.randint(0, len(agent)-1)
                b = random.randint(0, len(agent)-1)
                agent[a], agent[b] = agent[b], agent[a]
            new_agents.append(agent)
        return new_agents
    
    agents = [random.sample(jobs, len(jobs)) for _ in range(pop_size)]
    
    best_score_history = list()
    best_score = -1e9
    for i in range(100):
        agents = evaluate(agents)
        agents, (best_agent_res, best_agent, realisable) = select(agents)
        agents = crossover(agents)
        agents = mutate(agents)
        if (not realisable):
            raise ValueError("Yoo an imaginary gene.")
        agents = agents + [random.sample(jobs, len(jobs)) for _ in range(pop_size)]
        agents = agents[:pop_size]
        if (best_score < best_agent):
            best_score = best_agent
            print("Best score: {} GEN {}".format(best_score, i+1))
        best_score_history.append((best_score, i))
    return best_score
    # drawHeuristic(best_agent_res, data, block= False)
    # plt.figure()
    # plt.plot(range(len(best_score_history)), best_score_history)
    # plt.show(block= False)

def simulate(number_of_runs= 100, number_of_jobs= 5):
    h = list()
    for i in range(number_of_runs):
        print("Iter {}".format(i))
        data = generate_data(number_of_jobs) # make sure that this data is constant
        bres, (bscore, _) = branchAndBound(data)
        sres, sscore = schedule(data)
        _, h_score = heurisitc([], data)
        h.append((sscore, bscore, h_score[0]))
        if (bscore < sscore):
            raise ValueError("Heuristic is worse")
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.title("Score comparaison between B&B and EDD")
    plt.plot(range(len(h)), [x[1] for x in h])
    plt.plot(range(len(h)), [x[0] for x in h])
    plt.ylim(min([x[0] for x in h] + [x[1] for x in h] + [x[2] for x in h]) -3, 3)
    plt.legend(['B&B score', 'EDD score'])
    plt.subplot(2, 1, 2)
    plt.title("Score comparaison between B&B and score with preemption")
    plt.plot(range(len(h)), [x[1] for x in h])
    plt.plot(range(len(h)), [x[2] for x in h])
    plt.ylim(min([x[0] for x in h] + [x[1] for x in h] + [x[2] for x in h]) -3, 3)
    plt.legend(['B&B score', 'Preemption score'])
    plt.show()

def run(number_of_jobs= 5):
    data = generate_data(number_of_jobs)
    bres, (bscore, _) = branchAndBound(data, v= True)
    sres, sscore = schedule(data)
    print("Schedule score: {}, b&b: {}".format(sscore, bscore))
    drawGantt(sres, data, block= False)
    drawHeuristic(bres, data)

def responseTimeBenchmark():
    # heuristic time ms
    # b&b time ms
    # genetic algorithm time ms
    for n in range(4, 10):
        data = generate_data(n)
        res, (hscore, realisable) = heurisitc([], data)
        print("Heuristic score {} {}".format(hscore, realisable))
        geneticAlgorithm(data, 50)
        input()
        bres, bscore = branchAndBound(data, v= True)
        drawHeuristic(bres, data, block= True)
        print("bscore {}".format(bscore[0]))
        input()

def antsFormation(data: list[Job]):
    PH_MIN = 0.1
    PH_MAX = 10
    ALPHA = 1
    BETA = 1
    ANTS = 50
    ITER_MAX = 200

    jobs = [Job((t.r, t.p, t.d, t.id)) for t in data]
    n = len(jobs)
    pheromone : list[list[float]] = [[1]*n for _ in range(n)]
    def advance(ants: list[list[Job]], pheromone):
        res : list[list[Job]] = list()
        for ant in ants:
            # ants chooses next job
            # get remaining jobs
            # remaining_jobs = list(filter(lambda x : not x.id in [i.id for i in ant], ant))
            remaining_jobs = [j for j in jobs if not j.id in [x.id for x in ant] ]
            # remaining_jobs= []
            # ant_jobs_id = [j.id for j in ant]
            # for j in jobs:
            #     if (j.id in ant_jobs_id):
            #         continue
            #     remaining_jobs.append(j)
            
            if (len(remaining_jobs) == 0):
                print("remaining jobs = 0")
                continue
            # print("Remaining jobs", remaining_jobs)
            # calculate probabilities
            remaining_jobs_id = [j.id for j in remaining_jobs]
            pheromone_values = [(pheromone[ant[-1].id][j.id], j, 1) for j in remaining_jobs]
            
            # why not appending a job at the beginning of the path. that way the starting point wouldn't matter.
            pheromone_values_0 = [(pheromone[j.id][ant[0].id], j, 0) for j in remaining_jobs]
            
            pheromone_values.extend(pheromone_values_0)
            s = 0
            for p, j, _ in pheromone_values:
                s += p**ALPHA * (1/(j.r + j.d))**BETA
            
            prob_pool = [((ph**ALPHA * (1/(j.r + j.d)) **BETA)/(s + 1e-20), j, wer) for ph, j, wer in pheromone_values]
            prob_pool = [(0.0, 0, 0)] + prob_pool
            for i in range(1, len(prob_pool)):
                prob_pool[i] = (prob_pool[i-1][0] + prob_pool[i][0], prob_pool[i][1], prob_pool[i][2])
            
            # input()
            previous_ant_len = len(ant)
            next_job_id = 0
            r = random.random()
            for i in range(1, len(prob_pool)):
                if (prob_pool[i-1][0] < r and r <= prob_pool[i][0]):
                    if (prob_pool[i][2] == 0): # append job at the start of the path
                        ant = [prob_pool[i][1]] + ant
                    else:
                        ant.append(prob_pool[i][1])
            if (len(ant) == previous_ant_len):
                raise ValueError(f"YOO ANTS NOT APPENDING A JOB {len(ant)} {previous_ant_len} {r} {prob_pool}")
            # ant.append(remaining_jobs[next_job_id])
            # print(len(remaining_jobs), next_job_id)
            
            # ant.append(remaining_jobs[next_job_id])
            res.append(ant)
        # print(res)
        return res
    
    def updatePheromone(ants, pheromone):
        for ant in ants:
            # print("add")
            ids = [j.id for j in ant]
            for i in range(1, len(ids)):
                pheromone[ids[i-1]][ids[i]] += 10/(abs(getScoreHeuristic(ant)[0])**2 + 1e-3)
                # from i to j has not the same effect as from j to i, you should not be adding it.
                # pheromone[ids[i]][ids[i-1]] += 1/(abs(getScoreHeuristic(ant)[0]) + 1e-9)
                # print("adding ", 1/(abs(getScoreHeuristic(ant)[0] + 1)))
        # print(pheromone)
        return pheromone
    
    def evaporatePheromone(pheromone):
        for i in range(len(pheromone)):
            for j in range(len(pheromone)):
                pheromone[i][j] *= 0.95
                if pheromone[i][j] < PH_MIN :
                    pheromone[i][j] = PH_MIN
                if pheromone[i][j] > PH_MAX :
                    pheromone[i][j] = PH_MAX
                pheromone[i][j] = round(pheromone[i][j], 6)
        return pheromone
    
    best_fitness = -1e9
    fitness_history = list()
    for i in range(ITER_MAX):
        ants = [[random.choice(data)] for _ in range(ANTS)]
        for _ in range(len(data)-1):
            # print("Before advance {} {}".format(ants, pheromone))
            ants = advance(ants, pheromone)
            # print("After advance {} {}".format(ants, pheromone))
        pheromone = updatePheromone(ants, pheromone)
        # print(pheromone)
        # input()
        # for a in ants:
        #     print(a)
        # print("After update PH {} {}".format(ants, pheromone))
        pheromone = evaporatePheromone(pheromone)
        
        # draw gantt of the best solution

        fitness = [getScoreHeuristic(ant)[0] for ant in ants]
        if not any([getScoreHeuristic(ant)[1] for ant in ants]):
            raise ValueError("Ants providing an imaginary solution.")
        if (best_fitness < max(fitness)):
            best_fitness = max(max(fitness), best_fitness)
            print("Ants found : {} ITER {}".format(best_fitness, i+1))
        fitness_history.append((sum(fitness)/len(fitness), i))
    best_solution = max(ants, key= lambda ant : getScoreHeuristic(ant)[0])
    print(f"ANTS best sol : {best_solution} Score : {getScoreHeuristic(best_solution)} Data : {data}")
    # drawHeuristic(best_solution, data)
    # print([ant[0] for ant in ants])
    # for ph in pheromone:
    #     print(ph)
    # plt.figure()
    # plt.plot(list(range(len(fitness_history))), [x[0] for x in fitness_history])
    # plt.show()
    return best_fitness

def antsTuning():
    """This function will evaluate ants score vs genetic score"""
    ants_score = list()
    genetic_score = list()
    # bb_score = list()
    for i in range(20):
        data = generate_data(10)
        ant, gene = antsFormation(data), geneticAlgorithm(data)
        ants_score.append((ant, i))
        genetic_score.append((gene, i))
        # bb_score.append((bb, i))
    # ants_score = [(ants_score[i-1][0], ants_score[i-1][1]) for i in range(1, len(ants_score)) for _ in range(ants_score[i][1]-ants_score[i-1][1])]
    # ants_score += [(ants_score[-1][0], 1000)]
    # for i in range(len(bb_score)):
    #     print(f"B&B {bb_score[i][0]} GEN {genetic_score[i][0]} ANT {ants_score[i][0]}")
    plt.figure(figsize=(10, 10))
    # plt.plot([t[1] for t in bb_score], [t[0] for t in bb_score], linewidth = 4)
    plt.plot([t[1] for t in ants_score], [t[0] for t in ants_score])
    plt.plot([t[1] for t in genetic_score], [t[0] for t in genetic_score])
    plt.legend(['ANTS score', 'GENETIC score'])
    plt.show()

def main():
    print("1. Simulation graph.\n2. Run once.\n3. Change data generation parameters.")
    t = int(input())
    if (t == 1):
        print("Input number of runs.")
        n = int(input())
        print("Input number of jobs.")
        j = int(input())
        simulate(number_of_runs= n, number_of_jobs= j)
    elif (t == 2):
        print("Input number of jobs.")
        j = int(input())
        run(j)
    elif (t == 3):
        global lam_global
        global process_time_range
        global due_date_range
        print("Input lambda for the poisson distribution.")
        lam_global = int(input())
        print("Input min value of process time.")
        p_min = int(input())
        print("Input max value of process time.")
        p_max = int(input())
        process_time_range = (p_min, p_max)

        print("Input min value of due date.")
        d_min = int(input())
        print("Input max value of due date.")
        d_max = int(input())
        due_date_range = (d_min, d_max)
    if (t == 3):
        print("1. Simulation graph.\n2. Run once.")
        t = int(input())
        if (t == 1):
            print("Input number of runs.")
            n = int(input())
            print("Input number of jobs.")
            j = int(input())
            simulate(number_of_runs= n, number_of_jobs= j)
        elif (t == 2):
            print("Input number of jobs.")
            j = int(input())
            run(j)

def geneticSolution():
    data = generate_data(9)
    res, (hscore, realisable) = heurisitc([], data)
    print("Heuristic score {} {}".format(hscore, realisable))
    geneticAlgorithm(data, 50)
    input()
    bres, bscore = branchAndBound(data, v= True)
    drawHeuristic(bres, data, block= True)
    print("bscore {}".format(bscore[0]))
    input()
    
if __name__ == '__main__' :
    # main()
    # data = generate_data(5)
    # antsFormation(data)
    # geneticAlgorithm(data)
    antsTuning()