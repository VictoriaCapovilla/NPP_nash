import time

import numpy as np

from fstpso import FuzzyPSO

from GA.Project.lower_level import LowerLevel


class FST_PSO:

    def __init__(self, instance, lower_eps=10**(-12), swarm_size=None, save=False):

        self.save = save
        self.instance = instance

        self.n_od = self.instance.n_od
        self.n_paths = self.instance.n_paths
        self.swarm_size = swarm_size

        self.M = (self.instance.travel_time[:, -1] * (
                1 + self.instance.alpha * (self.instance.n_users / self.instance.q_od) ** self.instance.beta)).max()

        self.lower = LowerLevel(self.instance, lower_eps, M=self.M, save=save)

        self.obj_val = 0
        if self.save:
            self.times = []

    def fitness_evaluation(self, individual):
        individual = np.transpose(np.reshape(np.repeat(np.array(individual), repeats=self.n_od),
                                             (self.n_od, self.n_paths)))
        return - self.lower.eval(np.array(individual))

    def run_FST_PSO(self, max_iter):
        if self.save:
            self.times.append(time.time())
        dims = self.n_paths
        FP = FuzzyPSO()
        FP.set_search_space([[0, self.M]] * dims)
        FP.set_fitness(self.fitness_evaluation)
        if self.swarm_size is not None:
            FP.set_swarm_size(self.swarm_size)
        result = FP.solve_with_fstpso(max_iter=max_iter)
        print("Best solution:", result[0])
        self.obj_val = np.abs(result[1])
        print("Whose fitness is:", self.obj_val)
        if self.save:
            self.times += self.lower.total_time
            self.times = np.array(self.times)
            self.times = list(self.times - self.times[0])