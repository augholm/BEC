import numpy as np
from datasets import load_dataset


class GraphMatrix():

    def __init__(self, filename):
        n_jobs, n_machines, M, P = load_dataset(filename)
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.n = self.n_machines * self.n_jobs
        self.M = M
        self.P = P

        self.G = self._make_initial_graph(n=n_jobs*n_machines)

    def _make_initial_graph(self, n):
        n = self.n

        vals = np.repeat(np.arange(n),n).reshape(n, n) + 2
        for j in range(n):
            vals[j,(j+1) % n] = j+1
        vals = vals.T % (n+1)
        vals[vals == 0] = 1
        vals = np.array(-vals,int)
        G = np.pad(vals, 1, 'constant', constant_values=0)

        return np.array(G,int)

    def get_predecessors(self, i):
        G = self.G
        if G[0,i] == G[i,0] == 0: return []
        first, last = G[i,0], G[0,i]

        idx = first
        L = [idx]
        while idx != last:
            idx = G[i, idx]
            L.append(idx)
        return L

    def get_successors(self, i):
        G, n = self.G, self.n
        if G[i,-1] == G[-1, i] == 0: return []
        first, last = G[i,-1], G[-1,i]
        idx = first
        L = [idx]
        while idx != last:
            idx = G[i, idx] - n
            L.append(idx)
        return L

    def get_unknown(self, i):
        G = self.G
        idx = -G[i,i]
        if idx == -i: return []
        L = [idx]
        while idx != -G[i,idx]:
            idx = -G[i,idx]
            L.append(idx)
        return L

    def add_successor(self, i, j):
        '''
        add j to successor of i
        '''
        if i == 0:
            raise Warning('Successor should not be added for i = 0')
            return
        G, n = self.G, self.n
        if G[i,-1] == 0: G[i,-1] = j
        if G[-1,i] != 0:
            k = G[-1,i]
            G[i,k] = n+j
        G[i,j] = j+n
        G[-1,i] = j
        return

    def add_predecessor(self, i, j):
        '''
        add j as a predecessor to i
        '''
        if i == 0:
            raise Warning('Predecessor should not be added for i = 0')
            return
        G = self.G
        if G[i,0] == 0: G[i,0] = j  # if no predecessor..
        if G[0,i] != 0:  # if there exists a `last` one:
            k = G[0,i]
            G[i,k] = j  # then point to the new last one
        G[i,j] = j  # points to myself
        G[0,i] = j  # last predecessor is j

    def remove_unknown(self, i, j):
        G = self.G
        if self.G[i, j] >= 0: return  # not in unknown...
        if G[i,i] == -j:
            if G[i,j] == -j: G[i,i] = -i
            else: G[i,i] = G[i,j]
        else:
            k = -G[i,i]
            while G[i,k] != -j:
                k = -G[i,k]
            if G[i,j] == -j: G[i,k] = -k
            else: G[i,k] = G[i,j]
