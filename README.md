# Single Machine Scheduling Problem

## Problem brief description
This problem consists of a list of jobs and one single machine that needs to execute all of these jobs once at a time. A job is described as a tuple of 4 integers (release_date, process_duration, due_date, job_id).
- Release_date: The instant when the job is available for execution.
- Process_duration: The time needed for the job to finish exectuion.
- Due_date: The job deadline.
- Joe_id: A unique identifier for each job.

The problem is that we need to execute these jobs with minimal lateness. Meaning, optimally each job should finish before its corressponding deadline.
This problem is considered as a NP-Hard problem because the number of acceptable solutions is n! (n factorial).

This repo provides a B&B solution to this problem. The B&B method will find the optimal solution with the help of a heuristic function.

The performance of each solution will be measured with Lmax sore. Lmax ∈ [-∞, 0]. 0 means all jobs have met their deadline and -n meaning there is at least a job that didn't meet the deadline by n units.

**Assumptions :**
- All jobs are generated with the poisson process before the execution of the algorithm, to mimic the behaviour of clients or rush hours where there is a spike in jobs count for example.

## First approach: Earliest Due Date (EDD)
Sort jobs with respect to due date and release date of each job. Acceptable solution but not optimal.
This approach will be used as the starting solution of the Branch and Bound method.

## Second approach: Branch and Bound (B&B)
This approach uses a heuritic function that determines whether or not to branch or to bound a node (a group of solutions).
The heuristic function will exploit the constraint of each job must be executed without interruption. It will ignore this constraint and hence produce a non acceptable solution but will result in a much better (close to 0) Lmax score.

Upon running singlemachine.py file you will be greeted with a list of choices either to run a simution n times or just a single run to visualize gantt diagrams.
![](readme_images/simulation_menu.png)

**A single run**
![](readme_images/one_run.png)

**Simulation**
As you can see here, the B&B score is worse or equal to the heuristic score and better or equal than the EDD score, which is expected.
![](readme_images/simulation_output.png)

## Meta-heuristic approach

### Genetic Algorithm (GA)
We adapted and improved the implementation of genetic algorithm for this problem. For instance, each agent has a solution which is a specific order of jobs. Then, first 20% of the best agents were served to develop offsprings. The algorithm is fast for large number of jobs and gives optimal solution for data that is tested with the branch&bound approach.
But to compare this approach we thought it is wise to implement another meta-heuristic approach which was the ant colony optimization.

### Ant Colony Optmization (ACO)
Following the basic implementation of the ACO algorithm we have made some heavy adjustments to suit this problem and to perform well. One of the major changes that we had to make is that the ant is able to append jobs either at the beginning or at the end of the solution. This change have led to a better convergence.

## Conclusion
At a smaller number of jobs, branch&bound approach yields an optimal solution at a reasnable time but when it comes to much higher number of jobs, branch&bound suffers and wouldn't be suitable for real world cases. To overcome this, we could think of a better heuristic which in turn will enhance branch&bound and make it more efficient. Meta-heuristic approaches handle this with no significant performance drop. Even though the ACO is not well tuned, it sometimes yields a better solution compared to the genetic algorithm's solution but genetic algorithm is significantly faster than the ACO.