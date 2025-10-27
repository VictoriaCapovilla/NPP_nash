from random import Random

import numpy as np
import random

import pandas as pd
import torch

import time

from GA.genetic_algorithm_torch import GeneticAlgorithmTorch
from GA.random_torch import RandomTorch
from Instance.instance import Instance

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

POP_SIZE = 128 # 128 256 512
GENERATIONS = 1000  # 10 50 100
LOWER_EPS = 10 ** (-3)
OFFSPRING_RATE = 0.5
MUTATION_RATE = 0.02
LOWER_MAX_ITER = 30000
MAX_NO_IMPROVE = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_RUN = 10

SAVE = False
SAVE_PROBS = False
VERBOSE = False
REUSE_PROBS = False

vanilla_df = None

df = []

for paths in [10, 20, 30]:
    for n_od in [10, 20, 30]:
        instance = Instance(n_paths=paths, n_od=n_od)
        for run in range(N_RUN):
            t = time.time()

            # vanilla genetic algorithm
            ga = GeneticAlgorithmTorch(instance, POP_SIZE, lower_eps=LOWER_EPS, lower_max_iter=LOWER_MAX_ITER,
                                            offspring_proportion=OFFSPRING_RATE, mutation_rate=MUTATION_RATE,
                                            device=device, save=SAVE, save_probs=SAVE_PROBS, reuse_probs=REUSE_PROBS)
            ga_df = ga.run(GENERATIONS, max_no_improve=MAX_NO_IMPROVE, run_number=run, vanilla_df=vanilla_df, verbose=VERBOSE)


            # vanilla genetic algorithm
            rand = RandomTorch(instance, POP_SIZE, lower_eps=LOWER_EPS, lower_max_iter=LOWER_MAX_ITER,
                                            offspring_proportion=OFFSPRING_RATE,
                                            device=device, save=SAVE, save_probs=SAVE_PROBS, reuse_probs=REUSE_PROBS)
            rand_df = rand.run(ga.iterations, run_number=run, vanilla_df=vanilla_df, verbose=VERBOSE)

            df.append([paths, n_od, run, ga.best_val, ga.times[-1], ga.iterations, sum(ga.lower.data_time), rand.best_val, rand.times[-1]])

            print(str(paths) + '_' + str(n_od), ': run', run, 'of', N_RUN, 'complete', ga.best_val, rand.best_val, ga.times[-1], rand.times[-1], ga.iterations, '\n\n')
        df_to_save = pd.DataFrame(df, columns=['paths', 'n_od', 'run', 'best_val', 'time', 'iterations', 'lower_total_time','rand_best_val', 'rand_time'])
        df_to_save.to_csv('Results/GA_RANDOM/random_comparison.csv', index=False)