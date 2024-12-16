import ast
import time

import numpy as np
import random

import pandas as pd

import matplotlib.pyplot as plt

from GA.Project.PSO.FST_PSO import FST_PSO
# from GA.Project.PSO.PSO import PSO
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

# PATHS = 5
# N_OD = 5
SWARM_SIZE = None      # PSO default 10, FST-PSO 5_5 default int(10 + 2*np.sqrt(PATHS))
ITERATIONS = 100
LOWER_EPS = 10**(-4)

N_RUN = 10

total_df = None
n = 0
tot_run = N_RUN*3

for i in [20, 56, 90]:
    instance = Instance(n_paths=i, n_od=i)
    for j in range(N_RUN):
        t = time.time()

        # fuzzy self-tuning PSO
        genetic_algorithm = FST_PSO(instance, swarm_size=SWARM_SIZE, lower_eps=LOWER_EPS, save=True)

        genetic_algorithm.run_FST_PSO(ITERATIONS)

        # classic PSO
        # genetic_algorithm = PSO(instance, lower_eps=LOWER_EPS, save=True)
        #
        # hist = genetic_algorithm.run_PSO(ITERATIONS, swarm_size=SWARM_SIZE, max_velocity=[[0.001, 1]])

        # creating dataframe
        data = {
            'time': time.time() - t,
            'upper_iter': ITERATIONS,
            'fitness': float(genetic_algorithm.obj_val),
            'best_individual': [genetic_algorithm.lower.data_individuals[-1]],
            'upper_time': [genetic_algorithm.times],
            'lower_time': [genetic_algorithm.lower.data_time],
            'lower_iter': [genetic_algorithm.lower.n_iter],
            'fit_update': [genetic_algorithm.lower.data_fit],
            'ind_update': [genetic_algorithm.lower.data_individuals],
            'n_paths': genetic_algorithm.n_paths,
            'n_od': genetic_algorithm.instance.n_od,
            'n_users': [genetic_algorithm.instance.n_users],
            'swarm_size': int(10 + 2*np.sqrt(i)),
            # 'c_soc': genetic_algorithm.c_soc,
            # 'c_cog': genetic_algorithm.c_cog,
            # 'w': genetic_algorithm.w,
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

        print('PSO run', n, 'of', tot_run, 'complete', '\ntime:', time.time() - t, '\nfitness:', genetic_algorithm.obj_val, '\n')
        n += 1

        total_df.to_csv(r'C:\Users\viki\Desktop\NPP\Results\10_4\PSOstudy', index=False)

df = pd.read_csv(r'C:\Users\viki\Desktop\NPP\Results\10_4\PSOstudy')

for i in range(df.shape[0]):
    x = to_matrix(df.upper_time[i])[1:]
    y = to_matrix(df.fit_update[i])
    plt.plot(x, y, label=str(i))
plt.title("Plotting FST-PSO")
plt.xlabel("time")
plt.ylabel("fitness")
plt.legend()
plt.show()
