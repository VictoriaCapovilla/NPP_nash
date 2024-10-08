import numpy as np
import random

from Instance.lower_level import Lower


class GeneticAlgorithm:

    def __init__(self, instance, pop_size, offspring_proportion=0.5, tp_costs=None, tfp_costs=None, costs=None,
                 scale_factor=100, lower_eps=10**(-12)):

        self.instance = instance
        self.n_paths = self.instance.n_paths

        self.pop_size = pop_size
        self.n_children = int(pop_size * offspring_proportion)
        self.mat_size = self.pop_size + self.n_children
        self.scale_factor = scale_factor

        self.population = np.zeros((self.mat_size, self.n_paths))
        self.population[:self.pop_size, :self.n_paths] = np.random.uniform(size=(self.pop_size, self.n_paths))*self.scale_factor

        self.vals = np.zeros(self.mat_size)
        self.p = [[] for _ in range(self.mat_size)]

        self.lower = Lower(self.instance, lower_eps)
        for i in range(self.pop_size):
            self.vals[i], p = self.lower.eval(self.population[i])
            self.p[i] = p

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
                self.vals[self.pop_size + i], pp = self.lower.eval(self.population[self.pop_size + i])
                self.p[self.pop_size + i] = pp

            fitness_order = np.argsort(-self.vals)
            self.population = self.population[fitness_order]
            self.p = [self.p[i] for i in fitness_order]
            self.vals = self.vals[fitness_order]
            print(self.vals[0])
        print('costs =\n', self.population[0] )
        print(self.p[0])
        print('fitness =\n', self.vals[0])
