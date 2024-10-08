import numpy as np
import random

import torch

from Instance.lower_level import Lower
from Instance.lower_level_torch import LowerTorch


class GeneticAlgorithmTorch:

    def __init__(self, instance, pop_size, offspring_proportion=0.5, tp_costs=None, tfp_costs=None, costs=None,
                 scale_factor=100, lower_eps=10**(-12), device = None):

        self.device = device
        self.instance = instance
        self.n_paths = self.instance.n_paths

        self.pop_size = pop_size
        self.n_children = int(pop_size * offspring_proportion)
        self.mat_size = self.pop_size + self.n_children
        self.scale_factor = scale_factor


        self.M = self.scale_factor

        # TO DO calcolare max per la distribuzione prezzi
        self.population = torch.rand(size=(self.mat_size, self.n_paths), device= self.device) * self.M
        self.parents_idxs = torch.tensor([(i, j) for j in range(self.pop_size) for i in range(j + 1, self.pop_size)],
                                 device=self.device)

        self.vals = torch.zeros(self.mat_size, device=self.device)
        self.mask = torch.zeros(self.n_paths * self.n_children, device=self.device, dtype=torch.bool)

        self.lower = LowerTorch(self.instance, lower_eps, mat_size=self.mat_size, device = device)
        # for i in range(self.pop_size):
        #     self.vals[i] = self.lower.eval(self.population[i])

        self.vals = self.lower.eval(self.population)

    def run(self, iterations):
        for _ in range(iterations):
            parents = self.parents_idxs[torch.randperm(self.parents_idxs.shape[0])[:self.n_children]]

            self.mask[torch.randperm(self.mask.shape[0])[:self.mask.shape[0]//2]] = True

            self.population[self.pop_size:] = \
                (self.population[parents[:self.n_children][:, 0]] * self.mask.view(self.n_children, -1)
                 + self.population[parents[:self.n_children][:, 1]] * (~self.mask.view(self.n_children, -1)))

            p = torch.rand(size=(self.n_children, self.n_paths), device=self.device)
            idxs = torch.argwhere(p < 0.02)
            idxs[:, 0] += self.pop_size

            self.population[idxs[:, 0], idxs[:, 1]] = torch.rand(size=(idxs.shape[0],), device=self.device) * self.M

            self.mask[:] = False

            # for i in range(self.n_children):
            #     pop_max = self.population.max()
            #     a, b = np.random.choice(range(self.pop_size), size=2, replace=False)
            #     self.population[self.pop_size + i] = self.population[b]
            #     indexes = np.random.choice(range(self.n_paths), size=self.n_paths//2, replace=False)
            #     self.population[self.pop_size + i, indexes] = self.population[a][indexes]
            #     for index in range(self.n_paths):
            #         p = random.random()
            #         if p < 0.02:
            #             self.population[self.pop_size + i, index] = random.uniform(0, pop_max)

            self.vals = self.lower.eval(self.population)
            fitness_order = np.argsort(-self.vals.to('cpu'))
            self.population = self.population[fitness_order]
            self.vals = self.vals[fitness_order]
            # print(self.vals[0])
        print('costs =\n', self.population[0] * self.scale_factor)
        print('fitness =\n', self.vals[0])
