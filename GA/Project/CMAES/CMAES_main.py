import ast
import time

import numpy as np
import random

import torch

import pandas as pd

import matplotlib.pyplot as plt

from GA.Project.CMAES.CMAES import CMAES
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
ITERATIONS = 100
LOWER_EPS = 10**(-4)
STD = 1.1

N_RUN = 6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

instance = Instance(n_paths=PATHS, n_od=N_OD)

total_df = None

for i in range(N_RUN):
    t = time.time()
    genetic_algorithm = CMAES(instance, lower_eps=LOWER_EPS, device=device, reuse_p=None)

    genetic_algorithm.run_CMA(ITERATIONS, STD)

    # creating dataframe
    data = {
        'time': time.time() - t,
        'upper_iter': ITERATIONS,
        'fitness': float(genetic_algorithm.lower.data_fit[-1]),
        'best_individual': [genetic_algorithm.lower.data_individuals[-1]],
        'lower_time': [genetic_algorithm.lower.data_time],
        'lower_iter': [genetic_algorithm.lower.n_iter],
        'fit_update': [genetic_algorithm.lower.data_fit],
        'ind_update': [genetic_algorithm.lower.data_individuals],
        'std': STD,
        'n_paths': genetic_algorithm.n_paths,
        'n_od': genetic_algorithm.instance.n_od,
        'n_users': [genetic_algorithm.instance.n_users],
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

    print('Fitness:', float(genetic_algorithm.lower.data_fit[-1]))
    print('Execution time:', time.time() - t, '\nRun', i, 'complete', '\n')

total_df.to_csv(r'C:\Users\viki\Desktop\NPP\Results\10_4\CMA1_1', index=False)
df = pd.read_csv(r'/Results/10_4/CMA1_1')

x = np.array(range(0, 1000))
y = to_matrix(df.fit_update[0])[0:]
plt.title("Plotting CMA-ES")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.show()