import numpy as np
import util


def load_dataset(path):
    '''
    path: string that gives the path of the dataset to open

    returns:
        (n_jobs, n_machines, M, P)

        n_jobs, n_machines: int
        M: np.array of shape (n_jobs, n_machines)
            M[i,j] is the jth machine required for job i (in sequence)
        P: np.array of shape (n_jobs, n_machines)
            P[i,j] is the time spent on jth machine for job i

    '''
    L = []
    with open(path) as f:
        n_jobs, n_machines = util.extract_nums(f.readline(), int)

        for line in f.readlines():
            job = util.extract_nums(line, int)
            if len(job) != n_machines * 2:  # one per machine, one per time spent
                continue
            L.append(job)

    L = np.array(L)

    job_idx = np.arange(0, 2*n_machines, 2)
    time_idx = np.arange(1, 2*n_machines, 2)

    M, P = L[:, job_idx], L[:, time_idx]
    return n_jobs, n_machines, M, P
