import numpy as np
from forager import Forager
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.ion()



'''Profitability rating of honey patch depends on
    - distance from hive
    - nectar quality
    - nectar bounty


    Feedback (profitability rating) sets response threshold for an enlisting signal
    known as waggle dance. Length of waggle dance depends on response threshold
    and profitability rating of the patch

    Waggle dance is performed on (dance floor) where individual FORAGERS observe.
    Foragers can RANDOMLY select a dance to observe which gives location of the patch.

    '''


class BeeColony():
    def __init__(self, filename, n_foragers):
        self.foragers = []

        for k in range(n_foragers):
            self.foragers.append(Forager(filename))

        n = self.foragers[-1].n
        self.n = n
        #self.reset_foragers()

        self.colony_profitability_rating = 0

        self.best_foragers = []


    #don't think this applies - bees aren't supposed to be reset since there is no pheromones here
'''
    def reset_foragers(self):
        n = self.foragers[-1].n

        for F in self.foragers:
            F.current_op = 0
            F.S = np.arange(1, n, A.n_machines).tolist()
            F.T = []
            F.V = np.zeros(shape=self.n+2, dtype=int)  # the visited set
'''

    def populate_best_foragers(self, n_best):
        '''
        :param n_best: number of foragers to do the waggle dance

        updates best_foragers to the foragers with the best profitability ratings
        '''
        self.foragers.sort(key=lambda x: x.profitability_rating, reverse=True)

        if n_best <= len(self.foragers):
            for i in range(n_best):
                self.best_foragers.append(self.foragers[i])



    def get_shortest_path_length(self):
        return min([(each, each.calculate_tour_length()) for each in self.foragers],
                   key=lambda x: x[1])

    def update_colony_profitability_rating(self):
        total_profitability_rating = 0
        for forager in self.foragers:
            total_profitability_rating += forager.profitability_rating

        n_dancers = len(self.foragers)  # number of foragers doing the waggle dance, assume all foragers for now

        self.colony_profitability_rating = float(total_profitability_rating)/n_dancers




    def assign_follow_probability(self):
        '''
         Assigns probabilityes, r, to the foragers not in the best_foragers list

        '''
        for forager in self.foragers:
            forager_profitability = forager.profitability_rating
            if forager not in self.best_foragers:
                if forager_profitability < 0.9*self.colony_profitability_rating:
                    forager.r = 0.6
                elif forager_profitability < 0.95*self.colony_profitability_rating:
                    forager.r = 0.2
                elif forager_profitability < 1.15*self.colony_profitability_rating:
                    forager.r = 0.02
                else:
                    forager.r = 0.00

    def waggle_dance(self):




    def __repr__(self):
        return f'Bee Colony with {len(self.foragers)} foragers'


    def run(self, n_steps=10, plot=False):
        '''
        n_steps: int, number of steps to simulate

        returns best_forager, path_len
            best_forager: Forager instance
            path_len: int, the shortest path length

        '''
        L = []

        for step in tqdm(range(n_steps)):
            for t in range(self.n):
                self.step()
            for F in self.foragers:
                F.T.append(F.current_op)
            self.update_colony_profitability_rating()

            best_forager, path_len = self.get_shortest_path_length()
            L.append(path_len)

            if not step == n_steps - 1:
                self.reset_foragers()

        if plot:
            ax = plt.gca()
            ax.cla()
            ax.plot(L), ax.set_title('makespan vs. n_steps')

        return best_forager, path_len
