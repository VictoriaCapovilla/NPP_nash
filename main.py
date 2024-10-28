import time

import numpy as np
import random

import torch

import pandas as pd
from openpyxl import load_workbook

from GA.genetic_algorithm_torch import GeneticAlgorithmTorch
from Instance.instance import Instance
from old.genetic_algorithm import GeneticAlgorithm

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
    # print(time.time() - t, genetic_algorithm.obj_val)

    # creating dataframe
    data = {
                'time': genetic_algorithm.lower.data_time,
                'fitness': genetic_algorithm.data_fit,
                'best individual': genetic_algorithm.data_individuals,
                'probs_od1': genetic_algorithm.lower.data_probs,
                'n_iter': genetic_algorithm.lower.n_iter,
                'n_paths': [genetic_algorithm.n_paths for _ in range(ITERATIONS + 1)],
                'n_od': [genetic_algorithm.instance.n_od for _ in range(ITERATIONS + 1)],
                'pop_size': [genetic_algorithm.pop_size for _ in range(ITERATIONS + 1)],
                'alpha': [genetic_algorithm.instance.alpha for _ in range(ITERATIONS + 1)],
                'beta': [genetic_algorithm.instance.beta for _ in range(ITERATIONS + 1)],
                'M': [genetic_algorithm.M for _ in range(ITERATIONS + 1)],
                'K': [float(genetic_algorithm.lower.K) for _ in range(ITERATIONS + 1)],
                'run': [i for _ in range(ITERATIONS + 1)]
            }

    df = pd.DataFrame(data=data)

    if total_df is None:
        total_df = df
    else:
        total_df = pd.concat([total_df, df])

    print('run', i, 'complete', '\ntime:', time.time() - t)

total_df.to_csv(r"Results/output", index=False)
# print('DataFrame:\n', total_df)
