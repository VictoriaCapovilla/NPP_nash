import time

import numpy as np
import pandas as pd

import torch

from GA.lower_level_torch import LowerTorch


class RVGA_Gaussian:

    def __init__(self, instance, pop_size, offspring_proportion=0.5, lower_eps=10 ** (-12), lower_max_iter=30000,
                 device=None, save=False, save_probs=False, reuse_probs=False):

        self.save = save
        self.device = device
        self.instance = instance

        # network data
        self.n_paths = self.instance.n_paths

        self.pop_size = pop_size
        self.n_children = int(pop_size * offspring_proportion)
        self.n_parents = self.n_children * 2
        self.mat_size = self.pop_size + self.n_children

        # calculate individuals maximum value
        self.M = (self.instance.travel_time[:, -1] * (
                1 + self.instance.alpha * (self.instance.n_users / self.instance.q_od) ** self.instance.beta)).max()

        # initialize the Lower Level
        self.lower = LowerTorch(self.instance, lower_eps, mat_size=self.mat_size, device=device, M=self.M,
                                lower_max_iter=lower_max_iter, save=save, save_probs=save_probs,
                                reuse_probs=reuse_probs)

        # population initialization
        self.population = torch.rand(size=(self.mat_size, self.n_paths), device=self.device) * self.M
        self.pop_idxs = torch.tensor([i for i in range(self.mat_size)], device=self.device)

        # fitness evaluation
        self.vals = torch.zeros(self.mat_size, device=self.device)
        self.vals = self.lower.eval(self.population)

        if self.save:
            self.data_fit = []
            self.data_fit.append(float(self.vals[np.argsort(-self.vals.to('cpu'))][0]))

            self.data_individuals = []
            self.data_individuals.append(self.population[np.argsort(-self.vals.to('cpu'))][0].detach().cpu().numpy())

            self.times = []

    def tournament_selection(self, pop):
        # select randomly n_parents individuals from the pop
        tournament = torch.randperm(self.pop_idxs.shape[0], device=self.device)[:self.n_parents]
        # find the individual with the max fit among the selected
        fit = self.vals[tournament]
        selected = pop[torch.argmax(fit)]
        return selected

    def crossover(self, parents):
        # intermediate recombination crossover
        weights = torch.rand(size=(self.n_children, self.n_paths), device=self.device)
        children = weights * parents[:self.n_children] + (1 - weights) * parents[self.n_children:]
        return children

    def mutation(self, pop):
        # calculate std and mean of each individual
        std, mean = torch.std_mean(pop, dim=1)
        # create gaussian noise tensor
        noise = torch.repeat_interleave((torch.normal(mean, std)).unsqueeze(0), repeats=self.instance.n_paths, dim=0).T
        mutation = pop + noise
        # bounds check
        mutation = torch.where(mutation < 0, 0, mutation)
        mutation = torch.where(mutation > self.M, self.M, mutation)
        return mutation

    def run_gm(self, n_gen, run_number=None, gaussian_df=None):
        if self.save:
            self.times.append(time.time())
        for _ in range(n_gen):
            # TOURNAMENT SELECTION
            selected = torch.stack([self.tournament_selection(self.population) for _ in range(0, self.n_parents)])

            # CROSSOVER:
            self.population[self.pop_size:] = self.crossover(selected)

            # GAUSSIAN MUTATION
            self.population[self.pop_size:] = self.mutation(self.population[self.pop_size:])

            # FITNESS EVALUATION
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
            self.times = list(abs(self.times - self.times[0]))
            if gaussian_df is not False:
                gaussian_df = self.update_csv(n_gen, run_number, gaussian_df)
                return gaussian_df

    def update_csv(self, n_generations, run_number, gaussian_df):
        # creating dataframe
        data = {
            'time': self.times[-1],
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

        if gaussian_df is None:
            gaussian_df = df
        else:
            gaussian_df = pd.concat([gaussian_df, df])

        return gaussian_df
