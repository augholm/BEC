import numpy as np
from forager import Forager
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import copy

plt.ion()



class BeeColony():
    def __init__(self, filename, n_foragers):
        self.foragers = []

        for k in range(n_foragers):
            self.foragers.append(Forager(filename))

        n = self.foragers[-1].n
        self.n = n
        self.colony_profitability_rating = 0
        self.best_foragers = []
        self.alpha = 1
        self.beta = 1

        self.reset_foragers()

    def reset_foragers(self):
        n = self.foragers[-1].n


        for F in self.foragers:
            F.current_op = 0
            F.S = np.arange(1, n, F.n_machines).tolist()
            F.profitability_rating = 0
            F.global_op_count = 0
            F.r = 0
            # if F not in self.best_foragers:
            #     F.preferred_tour_indicator = 0
            #     F.preferred_tour = []
            # else:
            #     F.preferred_tour_indicator = 1
            #     F.preferred_tour = copy.deepcopy(F.T)
            F.T = []


    def step(self):
        '''
        Runs a single step
        - one step for each ant
        - in the end, the local step update is done (if True)
        '''

        if len(self.foragers[0].S) == 0:
            raise Warning('ants probably reached end..')

        for F in self.foragers:
            j = F.select_next_op(self.alpha, self.beta)
            F.move(j)


    def populate_best_foragers(self, n_best):
        '''
        :param n_best: number of foragers to do the waggle dance

        updates best_foragers to the foragers with the best profitability ratings
        '''
        self.foragers.sort(key=lambda x: x.profitability_rating, reverse=True)

        if n_best <= len(self.foragers):
            for i in range(n_best):
                self.best_foragers.append(self.foragers[i])
                self.foragers[i].preferred_tour = copy.deepcopy(self.foragers[i].T)
                self.foragers[i].preferred_tour_indicator = 1
        else:
            print("n_best larger than forager population")

    def get_shortest_path_length(self):
        return min([(each, each.calculate_tour_length()) for each in self.foragers],
                   key=lambda x: x[1])

    def update_colony_profitability_rating(self):
        total_profitability_rating = 0.0
        for forager in self.best_foragers:
            total_profitability_rating += forager.profitability_rating

        n_dancers = len(self.best_foragers)  # number of foragers doing the waggle dance, assume all foragers for now

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

    def waggle_dance(self, scaling_factor):

        # max_duration = scaling_factor*(self.best_foragers[0].profitability_rating / self.colony_profitability_rating)
        # min_duration = scaling_factor*(self.best_foragers[-1].profitability_rating / self.colony_profitability_rating)

        total_duration = 0
        durations = []

        for forager in self.best_foragers:
            duration = forager.profitability_rating / self.colony_profitability_rating
            durations.append(duration)
            total_duration+=duration
        durations = np.array(durations)
        durations = durations/total_duration

        for forager in self.foragers:
            if forager not in self.best_foragers:
                follow = random.random()
                print(forager.r)
                if follow <= forager.r:
                    print("YO")
                    forager.preferred_tour_indicator = 1
                    pick = np.random.choice(self.best_foragers, p=durations)
                    forager.preferred_tour = copy.deepcopy(pick.T)

        # for forager in self.foragers:
        #     follow = random.random()
        #     if ((follow <= forager.r) and (forager not in self.best_foragers)):
        #         forager.preferred_tour_indicator = 1  # Set bee as follower bee
        #         pick = random.uniform(min_duration, max_duration)
        #         current = 0
        #         for best_forager in self.best_foragers[::-1]:
        #             current += (best_forager.profitability_rating / self.colony_profitability_rating)
        #             if current > pick:
        #                 #  set path of this bee to the tour of best_forager bee
        #                 forager.preferred_tour = copy.deepcopy(best_forager.T)

    def __repr__(self):
        return f'Bee Colony with {len(self.foragers)} foragers'


    def run(self, n_steps=10, plot=False, n_best=20, scaling_factor=100):
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
                F.update_profitability_rating()

            best_forager, path_len = self.get_shortest_path_length()
            L.append(path_len)

            self.populate_best_foragers(n_best)
            self.update_colony_profitability_rating()
            self.assign_follow_probability()
            #self.populate_best_foragers(n_best)
            self.waggle_dance(scaling_factor)



            if not step == n_steps - 1:
                self.reset_foragers()

        if plot:
            ax = plt.gca()
            ax.cla()
            ax.plot(L), ax.set_title('makespan vs. n_steps')

        return best_forager, path_len
