import time

import numpy as np
import pandas as pd

from fstpso import FuzzyPSO

from GA.Project.lower_level import LowerLevel


class FST_PSO:

    def __init__(self, instance, lower_eps=10**(-12), lower_max_iter=30000, save=False):

        self.save = save
        self.instance = instance

        # network data
        self.n_od = self.instance.n_od
        self.n_paths = self.instance.n_paths

        # calculate individuals maximum value
        self.M = (self.instance.travel_time[:, -1] * (
                1 + self.instance.alpha * (self.instance.n_users / self.instance.q_od) ** self.instance.beta)).max()

        # initialize the Lower Level
        self.lower = LowerLevel(self.instance, lower_eps, M=self.M, lower_max_iter=lower_max_iter)

        self.obj_val = 0

    def fitness_evaluation(self, individual):
        # adapting the individual to Lower Level requirements
        individual = np.transpose(np.reshape(np.repeat(np.array(individual), repeats=self.n_od),
                                             (self.n_od, self.n_paths)))
        # return a negative value of the fitness since FST-PSO works with minimization
        return - self.lower.eval(individual)

    def run_FST_PSO(self, n_gen, swarm_size=None, run_number=None, FSTPSOdf=None):
        if self.save:
            t = time.time()
        dims = self.n_paths
        FP = FuzzyPSO()
        FP.set_search_space([[0, self.M]] * dims)
        FP.set_fitness(self.fitness_evaluation)
        if swarm_size is not None:
            FP.set_swarm_size(swarm_size)
        if self.save:
            result = FP.solve_with_fstpso(max_iter=n_gen,
                                      dump_best_fitness='//home/capovilla/Scrivania/NPP_nash/Results/data_fit',
                                      dump_best_solution='/home/capovilla/Scrivania/NPP_nash/Results/data_ind')
        else:
            result = FP.solve_with_fstpso(max_iter=n_gen)

        print("FST-PSO best solution:", [result[0]])
        self.obj_val = np.abs(result[1])
        print("Whose fitness is:", self.obj_val)

        if self.save:
            t = time.time() - t
            if FSTPSOdf is not False:
                FSTPSOdf = self.update_csv(t, n_gen, swarm_size, run_number, FSTPSOdf)
                return FSTPSOdf

    def update_csv(self, t, n_generations, swarm_size, run_number, FSTPSOdf):
        data_fit = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/data_fit', names=['fitness'])
        columns = [i for i in range(self.n_paths)]
        data_individuals = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/data_ind',
                                       names=columns, delimiter='	')
        # creating dataframe
        data = {
            'time': t,
            'upper_iter': n_generations,
            'fitness': float(self.obj_val),
            'best_individual': [data_individuals[:].values[-1]],
            'upper_time': [[t/n_generations * i for i in range(n_generations + 2)]],
            'fit_update': [np.abs(data_fit['fitness'].values)],
            'ind_update': [data_individuals[:].values],
            'n_paths': self.n_paths,
            'n_od': self.instance.n_od,
            'n_users': [self.instance.n_users],
            'swarm_size': swarm_size,
            'alpha': self.instance.alpha,
            'beta': self.instance.beta,
            'M': self.M,
            'K': float(self.lower.K),
            'eps': self.lower.eps,
            'run': run_number
        }

        df = pd.DataFrame(data=data)

        if FSTPSOdf is None:
            FSTPSOdf = df
        else:
            FSTPSOdf = pd.concat([FSTPSOdf, df])

        return FSTPSOdf
