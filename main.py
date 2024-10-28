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

N_RUN = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

instance = Instance(n_paths=PATHS, n_od=N_OD)

total_data = []

for i in range(N_RUN):
    t = time.time()
    genetic_algorithm = GeneticAlgorithmTorch(instance, POP_SIZE, lower_eps=LOWER_EPS, offspring_proportion=OFFSPRING_RATE,
                                              device=device, reuse_p=None)
    genetic_algorithm.run(ITERATIONS)
    # print(time.time() - t, genetic_algorithm.obj_val)

    # creating dataframe
    genetic_algorithm.data.append(
        {
            'run': i,
            'total_time': time.time() - t
        }
    )

    total_data += genetic_algorithm.data

    print('run', i, 'complete')

path = r"Results\run.xlsx"

book = load_workbook(path)
writer = pd.ExcelWriter(path, engine='openpyxl')
writer.book = book

df = pd.DataFrame(total_data)
df.to_excel(writer, sheet_name='Sheet')
writer.close()

# print('DataFrame:\n', df, '\n', 'execution time:\n', time.time() - t)
