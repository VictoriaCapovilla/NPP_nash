import numpy as np
import random

import torch

from Instance.lower_level import Lower
from Instance.lower_level_torch import LowerTorch


class GeneticAlgorithmTorch:

    def __init__(self, instance, pop_size, offspring_proportion=0.5, tp_costs=None, tfp_costs=None, costs=None,
                 scale_factor=100, lower_eps=10**(-12)):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.instance = instance
        self.n_paths = self.instance.n_paths



        self.pop_size = pop_size
        self.n_children = int(pop_size * offspring_proportion)
        self.mat_size = self.pop_size + self.n_children
        self.scale_factor = scale_factor

        self.population = np.zeros((self.mat_size, self.n_paths))
        self.population[:self.pop_size, :self.n_paths] = np.random.uniform(size=(self.pop_size, self.n_paths))

        self.vals = np.zeros(self.mat_size)

        self.lower = LowerTorch(self.instance, lower_eps, mat_size=self.mat_size)
        # for i in range(self.pop_size):
        #     self.vals[i] = self.lower.eval(self.population[i])

        self.vals = self.lower.eval(self.population)

    def run(self, iterations):
        for _ in range(iterations):
            for i in range(self.n_children):
                pop_max = self.population.max()
                a, b = np.random.choice(range(self.pop_size), size=2, replace=False)
                self.population[self.pop_size + i] = self.population[b]
                indexes = np.random.choice(range(self.n_paths), size=self.n_paths//2, replace=False)
                self.population[self.pop_size + i, indexes] = self.population[a][indexes]
                for index in range(self.n_paths):
                    p = random.random()
                    if p < 0.02:
                        self.population[self.pop_size + i, index] = random.uniform(0, pop_max)
            self.vals = self.lower.eval(self.population)
            fitness_order = np.argsort(-self.vals)
            self.population = self.population[fitness_order]
            self.vals = self.vals[fitness_order]
            print(self.vals[0])
        print('costs =\n', self.population[0] * self.scale_factor)
        print('fitness =\n', self.vals[0])
