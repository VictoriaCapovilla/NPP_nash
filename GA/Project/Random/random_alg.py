import time

import numpy as np
import pandas as pd

import torch

from GA.lower_level_torch import LowerTorch


class GeneticAlgorithmRandom:

    def __init__(self, instance, pop_size, offspring_proportion=0.5, lower_eps=10 ** (-12),
                 device=None, save=False):

        self.save = save
        self.device = device
        self.instance = instance

        # network data
        self.n_paths = self.instance.n_paths

        self.pop_size = pop_size
        self.n_children = int(pop_size * offspring_proportion)
        self.mat_size = self.pop_size + self.n_children

        # calculate individuals maximum value
        self.M = (self.instance.travel_time[:, -1] * (
                1 + self.instance.alpha * (self.instance.n_users / self.instance.q_od) ** self.instance.beta)).max()

        # initialize the lower level
        self.lower = LowerTorch(self.instance, lower_eps, mat_size=self.mat_size, device=device, M=self.M, save=save)

        # population initialization
        self.population = torch.rand(size=(self.mat_size, self.n_paths), device=self.device) * self.M

        #fitness evaluation
        self.vals = self.lower.eval(self.population)
        fitness_order = np.argsort(-self.vals.to('cpu'))
        self.population = self.population[fitness_order]
        self.vals = self.vals[fitness_order]

        if self.save:
            self.data_fit = []
            self.data_fit.append(float(self.vals[np.argsort(-self.vals.to('cpu'))][0]))

            self.data_individuals = []
            self.data_individuals.append(self.population[np.argsort(-self.vals.to('cpu'))][0].detach().cpu().numpy())

            self.times = []

        self.obj_val = 0

    def run(self, n_gen, run_number=None, random_df=None):
        if self.save:
            self.times.append(time.time())

        for _ in range(n_gen):
            # random individuals substitute the worst ones in the population
            self.population[self.pop_size:] = (torch.rand(size=(self.n_children, self.n_paths), device=self.device)
                                               * self.M)

            # fitness evaluation
            self.vals = self.lower.eval(self.population)
            fitness_order = np.argsort(-self.vals.to('cpu'))
            self.population = self.population[fitness_order]
            self.vals = self.vals[fitness_order]

            if self.save:
                self.data_individuals.append(self.population[0].detach().cpu().numpy())
                self.data_fit.append(float(self.vals[0]))

        print("Best fitness is:", self.vals[0])

        if self.save:
            self.times += self.lower.total_time
            self.times = np.array(self.times)
            self.times = list(self.times - self.times[0])
            if random_df is not False:
                random_df = self.update_csv(n_gen, run_number, random_df)
                return random_df

    def update_csv(self, n_generations, run_number, random_df):
        # creating dataframe
        data = {
            'time': self.times[-1],
            'case': str(self.n_paths) + '_' + str(self.instance.n_od) + '_' + str(self.n_paths),
            'upper_iter': n_generations,
            'fitness': float(self.data_fit[-1]),
            'best_individual': [self.data_individuals[-1]],
            'upper_time': [self.times],
            'lower_time': [self.lower.data_time],
            'lower_iter': [self.lower.n_iter],
            'fit_update': [self.data_fit],
            'ind_update': [self.data_individuals],
            'n_paths': self.n_paths,
            'n_od': self.instance.n_od,
            'n_users': [self.instance.n_users],
            'pop_size': self.pop_size,
            'alpha': self.instance.alpha,
            'beta': self.instance.beta,
            'M': self.M,
            'K': float(self.lower.K),
            'eps': self.lower.eps,
            'run': run_number
        }

        df = pd.DataFrame(data=data)

        if random_df is None:
            random_df = df
        else:
            random_df = pd.concat([random_df, df])

        return random_df
