import ast
import time

import numpy as np
import random

import torch

import pandas as pd

import matplotlib.pyplot as plt

from GA.Project.CMAES.CMAES import CMAES
from GA.Project.lower_level import LowerLevel
from GA.genetic_algorithm_torch import GeneticAlgorithmTorch
from GA.lower_level_torch import LowerTorch
# from GA.RVGA_Uniform.rv_uniform import RVGA_Uniform
# from GA.RVGA_Gaussian.rv_gaussian import RVGA_Gaussian
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
POP_SIZE = 5 #int(10 + 2*np.sqrt(PATHS))
LOWER_EPS = 10**(-4)
GENERATIONS = 20
OFFSPRING_RATE = 0.5
MUTATION_RATE = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

instance = Instance(n_paths=PATHS, n_od=N_OD)

N_RUN = 30

SAVE = True
total_df = None
VERBOSE = False
n = 0

t = time.time()

# vanilla genetic algorithm
genetic_algorithm = GeneticAlgorithmTorch(instance, POP_SIZE, lower_eps=LOWER_EPS,
                                          offspring_proportion=OFFSPRING_RATE, mutation_rate=MUTATION_RATE, device=device,
                                          reuse_p=None, save=True)
best_fit = genetic_algorithm.run(GENERATIONS, verbose=VERBOSE)

cmaes = CMAES(instance, lower_eps=LOWER_EPS)
cmaes.run_CMA(GENERATIONS, pop_size=POP_SIZE, verbose=VERBOSE)


# x = np.random.uniform(low=0, high=150, size=(POP_SIZE, N_OD))
# lower_torch = LowerTorch(instance, eps=LOWER_EPS, mat_size=POP_SIZE, device=device, M=113, save=False)
# vals = lower_torch.eval(torch.tensor(x))
# print(vals)
#
# # [5411.3747, 6195.0278, 5854.5367, 6569.0846, 5095.6198, 1273.4881],
# lower = LowerLevel(instance, M=113, eps=LOWER_EPS)
# print([lower.eval(x) for x in x])

# real valued genetic algorithm with UNIFORM MUTATION
# genetic_algorithm = RVGA_Uniform(instance, POP_SIZE, lower_eps=LOWER_EPS,
#                                  offspring_proportion=OFFSPRING_RATE, device=device, reuse_p=None, save=SAVE)
# best_fit = genetic_algorithm.run_um(GENERATIONS)

# real valued genetic algorithm with GAUSSIAN MUTATION
# genetic_algorithm = RVGA_Gaussian(instance, POP_SIZE, lower_eps=LOWER_EPS,
#                                  offspring_proportion=OFFSPRING_RATE, device=device, reuse_p=None, save=SAVE)
# best_fit = genetic_algorithm.run_gm(GENERATIONS)
