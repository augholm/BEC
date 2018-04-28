from DGM import GraphMatrix
import numpy as np


'''
All bees leave hive and search for a path. Then they return and we keep for exapmle top 10%. Each bee follows one of these
other bees based on waggle dance.
'''

class Forager(GraphMatrix):
    def __init__(self, filename):
        GraphMatrix.__init__(self, filename)
        self.S = []  # the open set
        self.T = []  # the tour list of the bee
        self.V = np.zeros(shape=self.n+2, dtype=int)  # the visited set
        self.current_op = 0
        self.profitability_rating = 0
        self.preferred_tour = 0  # whether a forager follows another forager or not
        self.r = 1  # probability of following a waggle dance




    def move(self, j):
        '''
        j: operation
        moves from i == self.current_op to j

        moves from i to j and updates
        - remove j from Unk(i) (and vice versa)
        - add j to Suc(i)
        - add i to Pre(j)
        - add i to self.V (the visited_set)
        - append i to self.T (the tour list)
        - adds new op if j is not last operation in sequence
        - removes j in `self.S` (the open list)

        returns None
        '''
        i = self.current_op

        if i != 0:
            self.remove_unknown(i, j), self.remove_unknown(j, i)
            self.add_successor(i, j)
            self.add_predecessor(j, i)
        self.V[i] = 1
        self.T.append(i)
        self.current_op = j
        self.S.remove(j)

        # j in n+2 --> reduce by one and get the job_no and seq_no
        seq_no = (j-1) % self.n_machines
        if seq_no != self.n_machines - 1:
            self.S.append(j+1)  # it's not the last operation --> append


    def select_next_op(self, alpha, beta):
        '''
        p_ij is the rating of edge between i and j
        alpha is the value assigned to the preferred path
        Need to add heuristic to choose nodes
        :return:
        '''

        i = self.current_op
        k = len(self.S)
        m = self.preferred_tour

        p = []  # ratings of edges between current node and all available nodes
        P = []  # probability to branch from node i to j
        #heuristic = []   heuristic distance between node i and j - can use with preferred path?

        if k <= m:
            for each in self.S:
                p.append((1-m*alpha)/(k-m))

        else:
            for each in self.S:
                p.append(alpha)

        numerator = 0
        for j in range(len(self.S)):
            numerator += (p[j]**alpha)
        for j in range(len(self.S)):
            P.append((p[j]**alpha) / numerator)

    def calculate_tour_length(self): #Find makespan for current bee
        C = np.zeros([self.n_jobs, self.n_machines])
        ops = np.array(self.T)
        seq_nos, job_nos = (ops-1) % self.n_machines, (ops-1) // self.n_machines
        m_nos, m_times = self.M[job_nos, seq_nos], self.P[job_nos, seq_nos]

        for op, j_idx, s_idx, m_idx, T in zip(ops, job_nos, seq_nos, m_nos, m_times):
            if op == 0: continue
            earliest_start = max(C[:, m_idx].max(), C[j_idx, :].max())
            C[j_idx, m_idx] = earliest_start + T
        return C.max()

    def update_profitability_rating(self):
        '''
        updates profitability rating with the formula 1/C_max
        '''
        C_max = calculate_tour_length(self)
        self.profitability_rating = float(1)/C_max