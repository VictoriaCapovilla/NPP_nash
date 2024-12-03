import ast
import time

import numpy as np
import random

import torch

import pandas as pd

import matplotlib.pyplot as plt

from GA.genetic_algorithm_torch import GeneticAlgorithmTorch
from Instance.instance import Instance


# Convert the cleaned string to an array
def to_matrix(a):
    cleaned_string = (a.replace('\r\n', ' ').replace('\n', ' ').strip().replace('   ', ' ').replace('  ', ' ')
                      .replace('[ ', '[').replace(' ]', ']').replace(' ', ',').replace(',,', ',').replace(',,', ','))

    return np.array(ast.literal_eval(cleaned_string))


seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

PATHS = 10
N_OD = 4
POP_SIZE = 256
LOWER_EPS = 10**(-4)
ITERATIONS = 100
OFFSPRING_RATE = 0.5

N_RUN = 6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

instance = Instance(n_paths=PATHS, n_od=N_OD)

total_df = None

for i in range(N_RUN):
    t = time.time()
    genetic_algorithm = GeneticAlgorithmTorch(instance, POP_SIZE, lower_eps=LOWER_EPS,
                                              offspring_proportion=OFFSPRING_RATE, device=device, reuse_p=None)

    genetic_algorithm.run(ITERATIONS)

    # creating dataframe
    data = {
        'time': time.time() - t,
        # 'upper_iter': ITERATIONS,
        'fitness': float(genetic_algorithm.obj_val),
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
        'run': i
            }

    df = pd.DataFrame(data=data)

    if total_df is None:
        total_df = df
    else:
        total_df = pd.concat([total_df, df])

    print('run', i, 'complete', '\ntime:', time.time() - t, '\nfitness:', genetic_algorithm.obj_val)

total_df.to_csv(r'C:/Users/viki/Desktop/NPP/Results/output', index=False)
df = pd.read_csv('C:/Users/viki/Desktop/NPP/Results/output')

for i in range(N_RUN):
    x = to_matrix(df.total_time[i])[1:]
    y = to_matrix(df.fit_update[i])
    plt.plot(x, y, label=str(i))
plt.title("Plotting GATorch")
plt.xlabel("time")
plt.ylabel("fitness")
plt.legend()
plt.show()
