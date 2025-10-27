import time

import numpy as np
import pandas as pd
import torch

from GA.lower_level_torch import LowerTorch


class RandomTorch:

    def __init__(self, instance, pop_size, offspring_proportion=0.5, lower_eps=10**(-12), lower_max_iter=30000,
                 device=None, save=False, save_probs=False, reuse_probs=False):

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
        self.lower = LowerTorch(self.instance, lower_eps, mat_size=self.mat_size, device=device, M=self.M,
                                lower_max_iter=lower_max_iter, save=save, save_probs=save_probs, reuse_probs=reuse_probs)

        # population initialization
        self.population = torch.rand(size=(self.mat_size, self.n_paths), device=self.device) * self.M

        self.vals = None
        self.best_val = -1

        if self.save:
            self.data_fit = []
            self.data_individuals = []
        self.times = []

    def run(self, n_gen, run_number=None, vanilla_df=None, verbose=False):
        if self.save:
            self.times.append(time.time())

        for _ in range(n_gen):

            self.population[self.pop_size:] = torch.rand(size=(self.n_children, self.n_paths), device=self.device) * self.M


            # fitness evaluation
            self.vals = self.lower.eval(self.population)
            best_val = self.vals.max().item()
            if self.best_val * 0.999 < best_val:
                self.best_val = best_val

            if verbose:
                print(_, self.best_val)

            if self.save:
                self.data_individuals.append(self.population[0].detach().cpu().numpy())
                self.data_fit.append(float(self.vals[0]))

        self.times += self.lower.total_time + [time.time()]
        self.times = np.array(self.times)
        self.times = list(self.times - self.times[0])
        if self.save:

            if vanilla_df is not False:
                vanilla_df = self.update_csv(n_gen, run_number, vanilla_df)
                return vanilla_df
            else:
                return None
        else:
            return None

    def update_csv(self, n_generations, run_number, vanilla_df):
        # creating dataframe
        data = {
            'time': self.times[-1],
            'case': str(self.n_paths) + '_' + str(self.instance.n_od) + '_' + str(self.pop_size),
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

        if vanilla_df is None:
            vanilla_df = df
        else:
            vanilla_df = pd.concat([vanilla_df, df])

        return vanilla_df
