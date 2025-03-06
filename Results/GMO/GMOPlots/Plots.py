import ast

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt


# Convert the cleaned string to an array
def to_matrix(a):
    cleaned_string = (a.replace('\r\n', ' ').replace('\n', ' ').replace('array(', '').replace('np.float64(', '')
                      .replace('   ', ' ').replace('  ', ' ').replace('[ ', '[').replace(' ]', ']')
                      .replace('(', '').replace(')', '').replace(' ', ',').replace(',,', ',').replace(',,', ','))

    return np.array(ast.literal_eval(cleaned_string))


CMA = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/GMO/Instances/CMAEStest')
PSO = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/GMO/Instances/PSOtest')
FST = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/GMO/Instances/FSTPSOtest')
uni = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/GMO/Instances/uniform_test')
gauss = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/GMO/Instances/gaussian_test')
vanilla = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/GMO/Instances/vanilla_test')
rdm = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/GMO/Instances/randomtest')

FST.rename({'swarm_size': 'pop_size'}, axis=1, inplace=True)
PSO.rename({'swarm_size': 'pop_size'}, axis=1, inplace=True)

# BEST CASE FOR EACH ALG
for _ in [CMA, PSO, FST, uni, gauss, vanilla, rdm]:
    a = _.loc[_.fitness == max(_.fitness)].index
    if _ is PSO:
        x = to_matrix(_.upper_time[a[0]])
        y = to_matrix(_.fit_update[a[0]])
    else:
        x = to_matrix(_.upper_time[a[0]])[1:]
        y = to_matrix(_.fit_update[a[0]])
    plt.plot(x, y, label=str(0))
plt.title("Best cases\n" + str(_.n_paths[0]) + " paths, " + str(_.n_od[0]) + " ODs, pop_size = " + str(_.pop_size[0])
          + ", gen = " + str(_.upper_iter[0]))
plt.xlabel("Time")
plt.ylabel("Fitness")
plt.legend(['CMA-ES', 'PSO', 'FST-PSO', 'rv_uniform', 'rv_gauss', 'vanilla', 'random'])
plt.show()

# WORST CASE FOR EACH ALG
for _ in [CMA, PSO, FST, uni, gauss, vanilla, rdm]:
    a = _.loc[_.fitness == min(_.fitness)].index
    if _ is PSO:
        x = to_matrix(_.upper_time[a[0]])
        y = to_matrix(_.fit_update[a[0]])
    else:
        x = to_matrix(_.upper_time[a[0]])[1:]
        y = to_matrix(_.fit_update[a[0]])
    plt.plot(x, y, label=str(0))
plt.title("Worst cases\n" + str(_.n_paths[0]) + " paths, " + str(_.n_od[0]) + " ODs, pop_size = " + str(_.pop_size[0])
          + ", gen = " + str(_.upper_iter[0]))
plt.xlabel("Time")
plt.ylabel("Fitness")
plt.legend(['CMA-ES', 'PSO', 'FST-PSO', 'rv_uniform', 'rv_gauss', 'vanilla', 'random'])
plt.show()

# MEANS FOR EACH ALG
for _ in [CMA, PSO, FST, uni, gauss, vanilla, rdm]:
    a = len(to_matrix(_.fit_update[0]))
    b = len(_.fitness)
    time = []
    fit = []
    for j in range(a):
        x = []
        y = []
        for i in range(b):
            if _ is PSO:
                x.append(to_matrix(_.upper_time[i])[j])
            else:
                x.append(to_matrix(_.upper_time[i])[j + 1])
            y.append(to_matrix(_.fit_update[i])[j])
        time.append(np.mean(x))
        fit.append(np.mean(y))
    plt.plot(time, fit, label=str(0))
plt.title("Mean\n" + str(_.n_paths[0]) + " paths, " + str(_.n_od[0]) + " ODs, pop_size = " + str(_.pop_size[0])
          + " gen = " + str(_.upper_iter[0]))
plt.xlabel("Time")
plt.ylabel("Fitness")
plt.legend(['CMA-ES', 'PSO', 'FST-PSO', 'rv_uniform', 'rv_gauss', 'vanilla', 'random'])
plt.show()

# BOX PLOTS
plt.boxplot([CMA.fitness, PSO.fitness, FST.fitness, uni.fitness, gauss.fitness, vanilla.fitness, rdm.fitness])
plt.xticks([1, 2, 3, 4, 5, 6, 7], ['CMA-ES', 'PSO', 'FST-PSO', 'rv_uniform', 'rv_gauss', 'vanilla', 'random'])
plt.ylabel('fitness')
plt.title(str(CMA.n_paths[0]) + " paths, " + str(CMA.n_od[0]) + " ODs, pop_size = " + str(CMA.pop_size[0])
          + ", gen = " + str(CMA.upper_iter[0]))
plt.show()

# ALL CASES FOR ALG _
CMA.name = 'CMA-ES'
PSO.name = 'PSO'
FST.name = 'FST-PSO'
uni.name = 'rv_uniform'
gauss.name = 'rv_gaussian'
vanilla.name = 'vanilla'
rdm.name = 'random'

for _ in [CMA, PSO, FST, uni, gauss, vanilla, rdm]:
    if _ is PSO:
        for i in range(len(_.fitness)):
            x = to_matrix(_.upper_time[i])
            y = to_matrix(_.fit_update[i])
            plt.plot(x, y, label=str(0))
    else:
        for i in range(len(_.fitness)):
            x = to_matrix(_.upper_time[i])[1:]
            y = to_matrix(_.fit_update[i])
            plt.plot(x, y, label=str(0))
    plt.title(_.name + "\n" + str(_.n_paths[0]) + " paths, " + str(_.n_od[0]) + " ODs, pop_size = " + str(_.pop_size[0])
              + ", gen = " + str(_.upper_iter[0]))
    plt.xlabel("Time")
    plt.ylabel("Fitness")
    plt.show()
