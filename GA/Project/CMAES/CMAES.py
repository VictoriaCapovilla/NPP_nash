import time

import numpy as np
import torch

from cmaes import CMA

from GA.Project.lower_level import LowerLevel


class CMAES:

    def __init__(self, instance, lower_eps=10**(-12), device=None, reuse_p=False):

        self.device = device
        self.instance = instance

        self.n_paths = self.instance.n_paths

        self.M = (self.instance.travel_time[:, -1] * (
                1 + self.instance.alpha * (self.instance.n_users / self.instance.q_od) ** self.instance.beta)).max()

        self.lower = LowerLevel(self.instance, lower_eps, device=device, M=self.M, reuse_p=reuse_p)

        self.times = []

    def fitness_evaluation(self, population):
        population = torch.repeat_interleave(torch.from_numpy(np.array(population)).unsqueeze(0),
                                             repeats=self.instance.n_od, dim=0)
        return - self.lower.eval(torch.from_numpy(np.array(population)))

    def run_CMA(self, iterations, sigma):
        self.times.append(time.time())
        optimizer = CMA(mean=np.zeros(self.n_paths), sigma=sigma, bounds=(np.array([[0, self.M]] * self.n_paths)))
        all_solutions = []
        for generation in range(iterations):
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()  # every time we need to ask the optimizer to sample a new solution
                value = self.fitness_evaluation(x)  # compute the fitness
                solutions.append((x, value))  # generate a vector of pairs (solution, fitness value)
            optimizer.tell(solutions)  # we tell CMA-ES what are the solutions and the corresponding fitness
            all_solutions.append(solutions)
        self.times += self.lower.total_time
        self.times = np.array(self.times)
        self.times = self.times - self.times[0]
        return all_solutions
