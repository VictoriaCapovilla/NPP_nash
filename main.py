import ast
import time

import numpy as np
import random

import torch

import pandas as pd

import matplotlib.pyplot as plt

from GA.genetic_algorithm_torch import GeneticAlgorithmTorch
from GA.RVGA_Uniform.rv_uniform import RVGA_Uniform
from GA.RVGA_Gaussian.rv_gaussian import RVGA_Gaussian
from Instance.instance import Instance


# Convert the cleaned string to an array
def to_matrix(a):
    cleaned_string = (a.replace('\r\n', ' ').replace('\n', ' ').replace('array(', '').replace('np.float64(', '')
                      .strip().replace('   ', ' ').replace('  ', ' ').replace('[ ', '[').replace(' ]', ']')
                      .replace('(', '').replace(')', '').replace(' ', ',').replace(',,', ',').replace(',,', ','))

    return np.array(ast.literal_eval(cleaned_string))


seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

PATHS = 5
N_OD = 5
POP_SIZE = int(10 + 2*np.sqrt(PATHS))
LOWER_EPS = 10**(-4)
GENERATIONS = 400
OFFSPRING_RATE = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

instance = Instance(n_paths=PATHS, n_od=N_OD)

N_RUN = 30

SAVE = True
total_df = None
n = 0

for k in range(N_RUN):
    t = time.time()

    # vanilla genetic algorithm
    # genetic_algorithm = GeneticAlgorithmTorch(instance, POP_SIZE, lower_eps=LOWER_EPS,
    #                                           offspring_proportion=OFFSPRING_RATE, device=device,
    #                                           reuse_p=None, save=True)
    # best_fit = genetic_algorithm.run(GENERATIONS)

    # real valued genetic algorithm with UNIFORM MUTATION
    # genetic_algorithm = RVGA_Uniform(instance, POP_SIZE, lower_eps=LOWER_EPS,
    #                                  offspring_proportion=OFFSPRING_RATE, device=device, reuse_p=None, save=SAVE)
    # best_fit = genetic_algorithm.run_um(GENERATIONS)

    # real valued genetic algorithm with GAUSSIAN MUTATION
    genetic_algorithm = RVGA_Gaussian(instance, POP_SIZE, lower_eps=LOWER_EPS,
                                     offspring_proportion=OFFSPRING_RATE, device=device, reuse_p=None, save=SAVE)
    best_fit = genetic_algorithm.run_gm(GENERATIONS)

    if SAVE:
        # creating dataframe
        data = {
            'time': time.time() - t,
            # 'mutation_range': [genetic_algorithm.mutation_range],
            # 'std': [std],
            'upper_iter': GENERATIONS,
            'fitness': float(best_fit),
            'best_individual': [genetic_algorithm.data_individuals[-1]],
            'upper_time': [genetic_algorithm.times],
            'lower_time': [genetic_algorithm.lower.data_time],
            'lower_iter': [genetic_algorithm.lower.n_iter],
            'fit_update': [genetic_algorithm.data_fit],
            'ind_update': [genetic_algorithm.data_individuals],
            # 'probs': [genetic_algorithm.lower.data_probs],
            # 'payoffs': [genetic_algorithm.lower.data_payoffs],
            'n_paths': genetic_algorithm.n_paths,
            'n_od': genetic_algorithm.instance.n_od,
            'n_users': [genetic_algorithm.instance.n_users],
            'pop_size': genetic_algorithm.pop_size,
            'alpha': genetic_algorithm.instance.alpha,
            'beta': genetic_algorithm.instance.beta,
            'M': genetic_algorithm.M,
            'K': float(genetic_algorithm.lower.K),
            'eps': genetic_algorithm.lower.eps,
            'run': n
        }
        df = pd.DataFrame(data=data)

        if total_df is None:
            total_df = df
        else:
            total_df = pd.concat([total_df, df])


        total_df.to_csv(r'C:\Users\viki\Desktop\NPP\Results\test', index=False)

    print('RV_G run', n + 1, 'of', N_RUN, 'complete', '\ntime:', time.time() - t, '\nfitness:', best_fit)
    n +=1

df = pd.read_csv(r'C:\Users\viki\Desktop\NPP\Results\test')

print((max(df.fitness) - df.fitness[0])/df.fitness[0] * 100)

for l in range(3):
    for i in range(10 * l, N_RUN + 10 * l):
        x = to_matrix(df.total_time[i])[1:]
        y = to_matrix(df.fit_update[i])
        plt.plot(x, y, label=str(i))
    plt.title("Plotting rv_GA")
    plt.xlabel("time")
    plt.ylabel("fitness")
    plt.legend()
    plt.show()
