import numpy as np
import random

import torch

import time

from GA.genetic_algorithm_torch import GeneticAlgorithmTorch
from GA.Project.RVGA_Uniform.rv_uniform import RVGA_Uniform
from GA.Project.RVGA_Gaussian.rv_gaussian import RVGA_Gaussian
from GA.Project.CMAES.CMAES import CMAES
from GA.Project.PSO.PSO import PSO
from GA.Project.PSO.FST_PSO import FST_PSO
from Instance.instance import Instance

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

PATHS = 20
N_OD = 20
POP_SIZE = 64
GENERATIONS = 400
LOWER_EPS = 10 ** (-4)
OFFSPRING_RATE = 0.5
MUTATION_RATE = 0.02

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

instance = Instance(n_paths=PATHS, n_od=N_OD)

N_RUN = 10
SAVE = False

vanilla_df = None
uniform_df = None
gaussian_df = None
CMAESdf = None
PSOdf = None
FSTPSOdf = None

for run in range(N_RUN):
    t = time.time()

    # # vanilla genetic algorithm
    # vanilla = GeneticAlgorithmTorch(instance, POP_SIZE, lower_eps=LOWER_EPS,
    #                                 offspring_proportion=OFFSPRING_RATE, mutation_rate=MUTATION_RATE,
    #                                 device=device, save=SAVE)
    # vanilla_df = vanilla.run(GENERATIONS, run_number=run, vanilla_df=vanilla_df)

    # real valued genetic algorithm with UNIFORM MUTATION
    rvga_uniform = RVGA_Uniform(instance, POP_SIZE, lower_eps=LOWER_EPS, offspring_proportion=OFFSPRING_RATE,
                                device=device, save=SAVE)
    uniform_df = rvga_uniform.run_um(GENERATIONS, run_number=run, uniform_df=uniform_df)

    # # real valued genetic algorithm with GAUSSIAN MUTATION
    # rvga_gaussian = RVGA_Gaussian(instance, POP_SIZE, lower_eps=LOWER_EPS, offspring_proportion=OFFSPRING_RATE,
    #                               device=device, save=SAVE)
    # gaussian_df = rvga_gaussian.run_gm(GENERATIONS, run_number=run, gaussian_df=gaussian_df)

    # # CMA-ES
    # CMA_ES = CMAES(instance, lower_eps=LOWER_EPS, save=SAVE)
    #
    # CMAESdf = CMA_ES.run_CMA(GENERATIONS, pop_size=POP_SIZE, run_number=run, CMAESdf=CMAESdf)

    # # classic PSO
    # classic_PSO = PSO(instance, lower_eps=LOWER_EPS, save=SAVE)
    #
    # PSOdf = classic_PSO.run_PSO(GENERATIONS, swarm_size=POP_SIZE, run_number=run, PSOdf=PSOdf)

    # # fuzzy self-tuning PSO
    # FST_PSO = FST_PSO(instance, lower_eps=LOWER_EPS, save=SAVE)
    #
    # FSTPSOdf = FST_PSO.run_FST_PSO(GENERATIONS, swarm_size=POP_SIZE, run_number=run, FSTPSOdf=FSTPSOdf)

    print('Run', run, 'of', N_RUN, 'complete', '\ntime:', time.time() - t)

if SAVE:
    # vanilla_df.to_csv('/home/capovilla/Scrivania/NPP_nash/Results/vanilla_test', index=False)
    uniform_df.to_csv('/home/capovilla/Scrivania/NPP_nash/Results/uniform_test', index=False)
    # gaussian_df.to_csv('/home/capovilla/Scrivania/NPP_nash/Results/gaussian_test', index=False)
    # CMAESdf.to_csv('/home/capovilla/Scrivania/NPP_nash/Results/CMAEStest', index=False)
    # PSOdf.to_csv('/home/capovilla/Scrivania/NPP_nash/Results/PSOtest', index=False)
    # FSTPSOdf.to_csv('/home/capovilla/Scrivania/NPP_nash/Results/FSTPSOtest', index=False)
