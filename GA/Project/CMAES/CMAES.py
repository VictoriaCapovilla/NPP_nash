import time

import numpy as np
import pandas as pd

from cmaes import CMA

from GA.Project.lower_level import LowerLevel


class CMAES:

    def __init__(self, instance, lower_eps=10**(-12), save=False):

        self.save = save
        self.instance = instance

        # network data
        self.n_od = self.instance.n_od
        self.n_paths = self.instance.n_paths

        # calculate individuals maximum value
        self.M = (self.instance.travel_time[:, -1] * (
                1 + self.instance.alpha * (self.instance.n_users / self.instance.q_od) ** self.instance.beta)).max()

        # initialize the Lower Level
        self.lower = LowerLevel(self.instance, lower_eps, M=self.M)

        if self.save:
            self.times = []
            self.data_individuals = []
            self.data_fit = []


    def fitness_evaluation(self, individual):
        # adapting the individual to Lower Level requirements
        individual = np.transpose(np.reshape(np.repeat(np.array(individual), repeats=self.n_od),
                                             (self.n_od, self.n_paths)))
        # return a negative value of the fitness since CMA-ES works with minimization
        return - self.lower.eval(individual)

    def run_CMA(self, n_gen, pop_size, verbose=False, run_number=None, CMAESdf=None):
        if self.save:
            self.times.append(time.time())
        optimizer = CMA(mean=(self.M / 2 * np.ones(self.n_paths)), sigma=(self.M * 0.3),
                        bounds=(np.array([[0, self.M]] * self.n_paths)), population_size=pop_size)
        for generation in range(n_gen):
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()  # sample a new solution
                value = self.fitness_evaluation(x)
                solutions.append((x, value))
            optimizer.tell(solutions)
            ind, fit = min(solutions, key = lambda t: t[1])
            if verbose:
                print(generation, fit)
            if self.save:
                self.data_individuals.append(ind)
                self.data_fit.append(float(np.abs(fit)))
                self.times.append(time.time())

        if self.save:
            self.times = np.array(self.times)
            self.times = list(self.times - self.times[0])
            if CMAESdf is not False:
                CMAESdf = self.update_csv(n_gen, pop_size, run_number, CMAESdf)
                return CMAESdf


    def update_csv(self, n_generations, pop_size, run_number, CMAESdf):
        # creating dataframe
        data = {
            'time': self.times[-1],
            'upper_iter': n_generations,
            'fitness': float(self.data_fit[-1]),
            'best_individual': [self.data_individuals[-1]],
            'upper_time': [self.times],
            'fit_update': [self.data_fit],
            'ind_update': [self.data_individuals],
            'std0': self.M * 0.3,
            'n_paths': self.n_paths,
            'n_od': self.instance.n_od,
            'n_users': [self.instance.n_users],
            'pop_size': pop_size,
            'alpha': self.instance.alpha,
            'beta': self.instance.beta,
            'M': self.M,
            'K': float(self.lower.K),
            'eps': self.lower.eps,
            'run': run_number
        }

        df = pd.DataFrame(data=data)

        if CMAESdf is None:
            CMAESdf = df
        else:
            CMAESdf = pd.concat([CMAESdf, df])

        return CMAESdf
