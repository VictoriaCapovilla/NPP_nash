import time

import numpy as np

from fstpso import FuzzyPSO

from GA.Project.lower_level import LowerLevel


class FST_PSO:

    def __init__(self, instance, lower_eps=10**(-12), save=False):

        self.save = save
        self.instance = instance

        # network data
        self.n_od = self.instance.n_od
        self.n_paths = self.instance.n_paths

        # calculate individuals maximum value
        self.M = (self.instance.travel_time[:, -1] * (
                1 + self.instance.alpha * (self.instance.n_users / self.instance.q_od) ** self.instance.beta)).max()

        # initialize the Lower Level
        self.lower = LowerLevel(self.instance, lower_eps, M=self.M)

        self.obj_val = 0

    def fitness_evaluation(self, individual):
        # adapting the individual to Lower Level requirements
        individual = np.transpose(np.reshape(np.repeat(np.array(individual), repeats=self.n_od),
                                             (self.n_od, self.n_paths)))
        # return a negative value of the fitness since FST-PSO works with minimization
        return - self.lower.eval(individual)

    def run_FST_PSO(self, max_iter, swarm_size = None):
        dims = self.n_paths
        FP = FuzzyPSO()
        FP.set_search_space([[0, self.M]] * dims)
        FP.set_fitness(self.fitness_evaluation)
        if swarm_size is not None:
            FP.set_swarm_size(swarm_size)
        if self.save:
            result = FP.solve_with_fstpso(max_iter=max_iter,
                                      dump_best_fitness='//home/capovilla/Scrivania/NPP_nash/Results/5x5/data_fit',
                                      dump_best_solution='/home/capovilla/Scrivania/NPP_nash/Results/5x5/data_ind')
        else:
            result = FP.solve_with_fstpso(max_iter=max_iter)
        print("Best solution:", [result[0]])
        self.obj_val = np.abs(result[1])
        print("Whose fitness is:", self.obj_val)