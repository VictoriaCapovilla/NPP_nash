import time

import numpy as np

import torch

from GA.lower_level_torch import LowerTorch


class GeneticAlgorithmTorch:

    def __init__(self, instance, pop_size, offspring_proportion=0.5, mutation_rate=0.02, lower_eps=10**(-12),
                 device=None, reuse_p=False, save=False):

        self.save = save
        self.device = device
        self.instance = instance

        self.n_paths = self.instance.n_paths

        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.n_children = int(pop_size * offspring_proportion)
        self.mat_size = self.pop_size + self.n_children
        self.mask = torch.zeros(self.n_paths * self.n_children, device=self.device, dtype=torch.bool)

        # calculate individuals maximum value
        self.M = (self.instance.travel_time[:, -1] * (
                1 + self.instance.alpha * (self.instance.n_users / self.instance.q_od) ** self.instance.beta)).max()

        # initialize the lower level
        self.lower = LowerTorch(self.instance, lower_eps, mat_size=self.mat_size, device=device, M=self.M,
                                reuse_p=reuse_p, save=save)

        # initialization
        self.population = torch.rand(size=(self.mat_size, self.n_paths), device=self.device) * self.M
        self.parents_idxs = torch.tensor([(i, j) for j in range(self.pop_size) for i in range(j + 1, self.pop_size)],
                                         device=self.device)

        self.parents = self.parents_idxs[torch.randperm(self.parents_idxs.shape[0],
                                                        device=self.device)[:self.n_children]]

        self.vals = torch.zeros(self.mat_size, device=self.device)
        self.vals = self.lower.eval(self.population)

        if self.save:
            self.data_fit = []
            self.data_fit.append(float(self.vals[np.argsort(-self.vals.to('cpu'))][0]))

            self.data_individuals = []
            self.data_individuals.append(self.population[np.argsort(-self.vals.to('cpu'))][0].detach().cpu().numpy())

            self.times = []

        self.obj_val = 0

    def run(self, iterations, verbose=False):
        if self.save:
            self.times.append(time.time())

        for _ in range(iterations):
            # selection
            self.parents = self.parents_idxs[torch.randperm(self.parents_idxs.shape[0],
                                                            device=self.device)[:self.n_children]]

            self.mask[torch.randperm(self.mask.shape[0], device=self.device)[:self.mask.shape[0]//2]] = True

            # crossover
            self.population[self.pop_size:] = \
                (self.population[self.parents[:self.n_children][:, 0]] * self.mask.view(self.n_children, -1)
                 + self.population[self.parents[:self.n_children][:, 1]] * (~self.mask.view(self.n_children, -1)))

            # mutation:
            p = torch.rand(size=(self.n_children, self.n_paths), device=self.device)
            idxs = torch.argwhere(p < self.mutation_rate)
            idxs[:, 0] += self.pop_size

            self.population[idxs[:, 0], idxs[:, 1]] = torch.rand(size=(idxs.shape[0],), device=self.device) * self.M

            self.mask[:] = False

            # fitness evaluation
            self.vals = self.lower.eval(self.population)
            fitness_order = np.argsort(-self.vals.to('cpu'))
            self.population = self.population[fitness_order]
            self.vals = self.vals[fitness_order]

            if verbose:
                print(_, self.vals[0].item())

            if self.save:
                self.data_individuals.append(self.population[0].detach().cpu().numpy())
                self.data_fit.append(float(self.vals[0]))

        if self.save:
            self.times += self.lower.total_time
            self.times = np.array(self.times)
            self.times = list(self.times - self.times[0])

        return self.vals[0]
