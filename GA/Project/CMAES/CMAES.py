import time

import numpy as np

from cmaes import CMA

from GA.Project.lower_level import LowerLevel


class CMAES:

    def __init__(self, instance, lower_eps=10**(-12), pop_size=None, save=False):

        self.save = save
        self.instance = instance

        self.n_od = self.instance.n_od
        self.n_paths = self.instance.n_paths
        self.pop_size = pop_size

        self.M = (self.instance.travel_time[:, -1] * (
                1 + self.instance.alpha * (self.instance.n_users / self.instance.q_od) ** self.instance.beta)).max()

        self.lower = LowerLevel(self.instance, lower_eps, M=self.M, save=save)

        if self.save:
            self.times = []

    def fitness_evaluation(self, individual):
        individual = np.transpose(np.reshape(np.repeat(np.array(individual), repeats=self.n_od),
                                             (self.n_od, self.n_paths)))
        return - self.lower.eval(np.array(individual))

    def run_CMA(self, iterations):
        if self.save:
            self.times.append(time.time())
        # optimizer = CMA(mean=np.random.uniform(0, self.M, size=self.n_paths), sigma=(0.3 * self.M), bounds=(np.array([[0, self.M]] * self.n_paths)),
        #                 population_size=self.pop_size)
        optimizer = CMA(mean=np.zeros(self.n_paths), sigma=0.5,
                        bounds=(np.array([[0, self.M]] * self.n_paths)), population_size=self.pop_size)
        all_solutions = []
        for generation in range(iterations):
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()  # every time we need to ask the optimizer to sample a new solution
                value = self.fitness_evaluation(x)  # compute the fitness
                solutions.append((x, value))  # generate a vector of pairs (solution, fitness value)
            optimizer.tell(solutions)  # we tell CMA-ES what are the solutions and the corresponding fitness
            all_solutions.append(solutions)
        if self.save:
            self.times += self.lower.total_time
            self.times = np.array(self.times)
            self.times = list(self.times - self.times[0])
        return all_solutions
