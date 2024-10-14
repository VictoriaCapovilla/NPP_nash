import time

import numpy as np
import random

import torch

from Instance.genetic_algorithm_torch import GeneticAlgorithmTorch
from Instance.instance import Instance
from Instance.genetic_algorithm import GeneticAlgorithm
# from Instance

seed = 0
np.random.seed(seed)
random.seed(seed)

PATHS = 40
N_OD = 10
POP_SIZE = 100
LOWER_EPS = 10**(-4)
ITERATIONS = 10
OFFSPRING_RATE = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


t = time.time()
instance = Instance(n_paths=PATHS, n_od=N_OD)  # n_path = available toll-paths


genetic_algorithm = GeneticAlgorithm(instance, POP_SIZE, offspring_proportion=OFFSPRING_RATE, lower_eps=LOWER_EPS)
genetic_algorithm.run(ITERATIONS)
print(time.time() - t, genetic_algorithm.obj_val)
t = time.time()
genetic_algorithm = GeneticAlgorithmTorch(instance, POP_SIZE, lower_eps=LOWER_EPS, offspring_proportion=OFFSPRING_RATE,
                                          device=device)

genetic_algorithm.run(ITERATIONS)

print(time.time() - t, genetic_algorithm.obj_val)

import numpy as np

p = np.array(range(4))
g = p
g[0] = 9