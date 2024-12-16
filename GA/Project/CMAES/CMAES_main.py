import ast
import time

import numpy as np
import random

import pandas as pd

import matplotlib.pyplot as plt

from GA.Project.CMAES.CMAES import CMAES
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

PATHS = 5
N_OD = 5
POP_SIZE = None
ITERATIONS = 100
LOWER_EPS = 10**(-4)
# STD0 = 1.1

N_RUN = 10

instance = Instance(n_paths=PATHS, n_od=N_OD)

total_df = None
n = 0
tot_run = N_RUN * 3

for i in [20, 56, 90]:
    instance = Instance(n_paths=i, n_od=i)
    POP_SIZE = int(10 + 2*np.sqrt(i))
    for j in range(N_RUN):
        t = time.time()
        genetic_algorithm = CMAES(instance, pop_size=POP_SIZE, lower_eps=LOWER_EPS, save=True)

        genetic_algorithm.run_CMA(ITERATIONS)

        # creating dataframe
        data = {
            'time': time.time() - t,
            'upper_iter': ITERATIONS,
            'fitness': float(genetic_algorithm.lower.data_fit[-1]),
            'best_individual': [genetic_algorithm.lower.data_individuals[-1]],
            'upper_time': [genetic_algorithm.times],
            'lower_time': [genetic_algorithm.lower.data_time],
            'lower_iter': [genetic_algorithm.lower.n_iter],
            'fit_update': [genetic_algorithm.lower.data_fit],
            'ind_update': [genetic_algorithm.lower.data_individuals],
            'std0': 0.5,
            'n_paths': genetic_algorithm.n_paths,
            'n_od': genetic_algorithm.instance.n_od,
            'n_users': [genetic_algorithm.instance.n_users],
            'pop_size': POP_SIZE,
            'alpha': genetic_algorithm.instance.alpha,
            'beta': genetic_algorithm.instance.beta,
            'M': genetic_algorithm.M,
            'K': float(genetic_algorithm.lower.K),
            'eps': genetic_algorithm.lower.eps,
            'run': n
        }

        df = pd.DataFrame(data=data)

        if total_df is None:
            total_df = df
        else:
            total_df = pd.concat([total_df, df])

        print('Fitness:', float(genetic_algorithm.lower.data_fit[-1]))
        print('Execution time:', time.time() - t, '\nRun', n, 'of', tot_run, 'complete', '\n')

        n += 1

        total_df.to_csv(r'C:\Users\viki\Desktop\NPP\Results\CMAstudy', index=False)

df = pd.read_csv(r'C:\Users\viki\Desktop\NPP\Results\CMAstudy')

for i in range(df.shape[0]):
    x = to_matrix(df.upper_time[i])[1:]
    y = to_matrix(df.fit_update[i])
    plt.plot(x, y, label=str(i))
plt.title("Plotting CMA-ES")
plt.xlabel("time")
plt.ylabel("fitness")
plt.legend()
plt.show()
