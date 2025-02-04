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

# PATHS = 20
# N_OD = 20
POP_SIZE = 128
GENERATIONS = 2
LOWER_EPS = 10 ** (-4)
OFFSPRING_RATE = 0.5
MUTATION_RATE = 0.02

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_RUN = 3
SAVE = True

vanilla_df = None

for paths in [20, 56, 90]:
    for n_od in [20, 56, 90]:
        instance = Instance(n_paths=paths, n_od=n_od)
        for run in range(N_RUN):
            t = time.time()

            # vanilla genetic algorithm
            vanilla = GeneticAlgorithmTorch(instance, POP_SIZE, lower_eps=LOWER_EPS,
                                            offspring_proportion=OFFSPRING_RATE, mutation_rate=MUTATION_RATE,
                                            device=device, save=SAVE)
            vanilla_df = vanilla.run(GENERATIONS, run_number=run, vanilla_df=vanilla_df)

            vanilla_df.to_csv('/home/capovilla/Scrivania/NPP_nash/Results/test', index=False)
            print(str(paths) + '_' + str(n_od), ': run', run, 'of', N_RUN, 'complete', '\ntime:', time.time() - t)

# if SAVE:
    # vanilla_df.to_csv('/home/capovilla/Scrivania/NPP_nash/Results/vanilla_test', index=False)
