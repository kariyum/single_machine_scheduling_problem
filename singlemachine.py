import random
import numpy as np
import matplotlib.pyplot as plt


def generate_data(n):
    """
        n number of jobs
    """
    release_date = np.random.poisson(lam=12, size=n)
    process_time = [random.randint(1, 10) for _ in range(len(release_date))]
    due_date = [x[0] + x[1] + random.randint(0, 10) for x in zip(release_date, process_time)]
    jobs = [(d[0], d[1], d[2], id) for id, d in enumerate(zip(release_date, process_time, due_date))]
    return jobs

def schedule(data):
    # data contains : release date, due date, process time, jobID
    # sort by release date then sort by due date
    # data = [(r0, d0, p0, jobID), ..., (rj, dj, pj, jobID)...]
    sorted_data = sorted(data, key = lambda job : (job[0], job[1]))
    return sorted_data

if __name__ == '__main__' :
    data = generate_data(10)
    res = schedule(data)
    aux_res = res.copy()
    print(res)
    plt.figure(figsize = (10, 10))
    for i in range(1, len(res)):
        res[i] = (max(res[i-1][0] + res[i-1][1], res[i][0]), res[i][1], res[i][2], res[i][3])
    plt.subplot(2, 1, 1)
    plt.barh(y = [str(x[3]) for x in res], width = [x[1] for x in res], left= [x[0] for x in res])
    plt.title("Solution plot GANTT")
    plt.xlim([min([x[0] for x in aux_res]), max([max(res[i-1][0] + res[i-1][1], res[i][0]) + x[1] for x in aux_res])])
    plt.subplot(2, 1, 2)
    plt.barh(y = [str(x[3]) for x in aux_res], width = [x[1] for x in aux_res], left= [x[0] for x in aux_res])
    plt.xlim([min([x[0] for x in aux_res]), max([max(res[i-1][0] + res[i-1][1], res[i][0]) + x[1] for x in aux_res])])
    plt.title("Data plot GANTT")
    plt.show()
