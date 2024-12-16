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
        # self.mask = torch.zeros(self.n_paths * self.n_children, device=self.device, dtype=torch.bool)

        self.M = (self.instance.travel_time[:, -1] * (
                1 + self.instance.alpha * (self.instance.n_users / self.instance.q_od) ** self.instance.beta)).max()

        self.lower = LowerTorch(self.instance, lower_eps, mat_size=self.mat_size, device=device, M=self.M, reuse_p=reuse_p)

        # initialization
        self.population = torch.rand(size=(self.mat_size, self.n_paths), device=self.device) * self.M
        self.parents_idxs = torch.tensor([(i, j) for j in range(self.pop_size) for i in range(j + 1, self.pop_size)],
                                         device=self.device)

        self.parents = self.parents_idxs[torch.randperm(self.parents_idxs.shape[0])[:self.n_children]]

        # fitness evaluation
        self.vals = torch.zeros(self.mat_size, device=self.device)
        self.vals = self.lower.eval(self.population)

        self.data_fit = []
        self.data_fit.append(float(self.vals[np.argsort(-self.vals.to('cpu'))][0]))

        self.data_individuals = []
        self.data_individuals.append(self.population[0].detach().cpu().numpy())

        self.obj_val = 0

    def run(self, iterations):
        for _ in range(iterations):
            # SELECTION
            self.parents = self.parents_idxs[torch.randperm(self.parents_idxs.shape[0])[:self.n_children]]

            # CROSSOVER:
            weights = torch.rand(size=(self.n_children, self.n_paths), device=self.device)
            self.population[self.pop_size:] = \
                (weights * self.population[self.parents[:self.n_children][:, 0]]
                 + (1 - weights) * self.population[self.parents[:self.n_children][:, 1]])

            # GAUSSIAN MUTATION
            # mean = torch.mean(self.population[self.pop_size:])

            # sp = torch.mean((self.population[self.pop_size:])) ** 2     # Signal Power
            # snr = 18                                                    # Signal-to-noise ratio
            # std = (sp / snr) ** 0.5

            # ostd = 0.5 * torch.std(self.population[self.pop_size:])

            # std =

            noise = torch.normal(0, std)

            self.population[self.pop_size:] = self.population[self.pop_size:] + noise
            torch.where(self.population[self.pop_size:] < 0, 0, self.population[self.pop_size:])
            torch.where(self.population[self.pop_size:] > self.M, self.M, self.population[self.pop_size:])

            # # valutare mutazione su tutta la popolazione

            # FITNESS EVALUATION
            self.vals = self.lower.eval(self.population)
            fitness_order = np.argsort(-self.vals.to('cpu'))
            self.population = self.population[fitness_order]
            self.vals = self.vals[fitness_order]

            # # sistemare elitismo
            # idea: salvare i nuovi self.population[self.pop_size:] a parte e tenere i migliori self.mat_size tra
            # questi e la vecchia popolazione

            self.data_individuals.append(self.population[0].detach().cpu().numpy())
            self.data_fit.append(float(self.vals[0]))
            # print(self.vals[0])

        self.obj_val = self.vals[0]

        # print('costs =\n', self.population[0])
        # print('fitness =\n', self.vals[0])
