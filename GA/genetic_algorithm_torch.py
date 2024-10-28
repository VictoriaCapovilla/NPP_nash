import time

import numpy as np

import torch

from GA.lower_level_torch import LowerTorch


class GeneticAlgorithmTorch:

    def __init__(self, instance, pop_size, offspring_proportion=0.5, lower_eps=10**(-12), device=None, reuse_p=False):

        self.device = device
        self.instance = instance

        self.n_paths = self.instance.n_paths

        self.pop_size = pop_size
        self.n_children = int(pop_size * offspring_proportion)
        self.mat_size = self.pop_size + self.n_children
        self.mask = torch.zeros(self.n_paths * self.n_children, device=self.device, dtype=torch.bool)

        self.M = (self.instance.travel_time[:, -1] * (
                1 + self.instance.alpha * (self.instance.n_users / self.instance.q_od) ** self.instance.beta)).max()

        self.lower = LowerTorch(self.instance, lower_eps, mat_size=self.mat_size, device=device, M=self.M, reuse_p=reuse_p)

        self.population = torch.rand(size=(self.mat_size, self.n_paths), device=self.device) * self.M
        self.parents_idxs = torch.tensor([(i, j) for j in range(self.pop_size) for i in range(j + 1, self.pop_size)],
                                         device=self.device)

        self.parents = self.parents_idxs[torch.randperm(self.parents_idxs.shape[0])[:self.n_children]]

        self.vals = torch.zeros(self.mat_size, device=self.device)
        self.vals = self.lower.eval(self.population)

        self.data_fit = []
        self.data_fit.append(float(self.vals[0]))

        self.data_individuals = []
        self.data_individuals.append(self.population[0].detach().cpu().numpy())

        self.obj_val = 0

    def run(self, iterations):
        for _ in range(iterations):
            # crossover:
            # choose the parents
            self.parents = self.parents_idxs[torch.randperm(self.parents_idxs.shape[0])[:self.n_children]]

            self.mask[torch.randperm(self.mask.shape[0])[:self.mask.shape[0]//2]] = True

            # make the children
            self.population[self.pop_size:] = \
                (self.population[self.parents[:self.n_children][:, 0]] * self.mask.view(self.n_children, -1)
                 + self.population[self.parents[:self.n_children][:, 1]] * (~self.mask.view(self.n_children, -1)))

            # mutation:
            p = torch.rand(size=(self.n_children, self.n_paths), device=self.device)
            idxs = torch.argwhere(p < 0.02)
            idxs[:, 0] += self.pop_size

            self.population[idxs[:, 0], idxs[:, 1]] = (torch.rand(size=(idxs.shape[0],), device=self.device) *
                                                       self.M)

            self.mask[:] = False

            # fitness evaluation
            self.vals = self.lower.eval(self.population)
            fitness_order = np.argsort(-self.vals.to('cpu'))
            self.population = self.population[fitness_order]
            self.vals = self.vals[fitness_order]

            self.data_individuals.append(self.population[0].detach().cpu().numpy())
            self.data_fit.append(float(self.vals[0]))
            # print(self.vals[0])

        self.obj_val = self.vals[0]

        # print('costs =\n', self.population[0])
        # print('fitness =\n', self.vals[0])
