import numpy as np
import random

import torch

import time

from GA.genetic_algorithm_torch import GeneticAlgorithmTorch
from Instance.instance import Instance

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

POP_SIZE = 128
GENERATIONS = 1000
LOWER_EPS = 10 ** (-4)
OFFSPRING_RATE = 0.5
MUTATION_RATE = 0.02
LOWER_MAX_ITER = 30000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_RUN = 1
SAVE = True
SAVE_PROBS = False
VERBOSE = False
REUSE_PROBS = False
vanilla_df = None

for paths in [5]:
    for n_od in [5]:
        instance = Instance(n_paths=paths, n_od=n_od)
        for run in range(N_RUN):
            t = time.time()

            # vanilla genetic algorithm
            vanilla = GeneticAlgorithmTorch(instance, POP_SIZE, lower_eps=LOWER_EPS,
                                            offspring_proportion=OFFSPRING_RATE, mutation_rate=MUTATION_RATE, lower_max_iter=LOWER_MAX_ITER,
                                            device=device, save=SAVE, save_probs=SAVE_PROBS, reuse_probs=REUSE_PROBS)
            vanilla_df = vanilla.run(GENERATIONS, run_number=run, vanilla_df=vanilla_df, verbose=VERBOSE)

            if SAVE:
                vanilla_df.to_csv('/home/capovilla/Scrivania/NPP_nash/128_1000', index=False)

            print(str(paths) + '_' + str(n_od), ': run', run, 'of', N_RUN, 'complete', '\ntime:', time.time() - t)
