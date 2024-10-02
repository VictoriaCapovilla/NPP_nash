import numpy as np
import random

from Instance.genetic_algorithm_torch import GeneticAlgorithmTorch
from Instance.instance import Instance
from Instance.genetic_algorithm import GeneticAlgorithm
# from Instance

seed = 0
np.random.seed(seed)
random.seed(seed)

PATHS = 6
N_OD = 5
POP_SIZE = 10
LOWER_EPS = 10**(-4)
ITERATIONS = 200
OFFSPRING_RATE = 0.5
SCALE = 10

instance = Instance(n_paths=PATHS, n_od=N_OD, scale_factor=10)  # n_path = available toll-paths
genetic_algorithm = GeneticAlgorithm(instance, POP_SIZE, lower_eps=LOWER_EPS, offspring_proportion=OFFSPRING_RATE)
genetic_algorithm.run(ITERATIONS)
print(instance.tfp_costs)
