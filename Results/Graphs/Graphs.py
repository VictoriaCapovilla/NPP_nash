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


CMA = pd.read_csv(r'../20x20/CMA')
PSO = pd.read_csv(r'../20x20/PSO')
FST = pd.read_csv(r'../20x20/FSTPSO')
uni = pd.read_csv(r'../20x20/rv_uniform')
gauss = pd.read_csv(r'../20x20/rv_gauss')
torch = pd.read_csv(r'../20x20/torch')

# CMA32 = pd.read_csv(r'../5x5/3200/CMA')
# CMA56 = pd.read_csv(r'../5x5/5600/CMA')
# CMA80 = pd.read_csv(r'../5x5/8000/CMA')

# a = PSO.loc[PSO.fitness == max(PSO.fitness)].index
# print('PSO', a)
# print(a[0])
# print('time\n', to_matrix(PSO.upper_time[a[0]]))
# print('fit\n', to_matrix(PSO.fit_update[a[0]]))
# print(max(to_matrix(PSO.fit_update[a[0]])))

# CMA, PSO, FST, uni, gauss, torch
# 'CMA', 'PSO', 'FST-PSO', 'rv_uniform', 'rv_gauss', 'vanilla'

# BEST/WORST CASE FOR EACH ALG
# for _ in [CMA, PSO, FST, uni, gauss, torch]:
#     print(min(_.fitness))
#     a = _.loc[_.fitness == min(_.fitness)].index
#     if _ is PSO:
#         x = to_matrix(_.upper_time[a[0]])
#         y = to_matrix(_.fit_update[a[0]])
#     else:
#         x = to_matrix(_.upper_time[a[0]])[1:]
#         y = to_matrix(_.fit_update[a[0]])
#     plt.plot(x, y, label=str(0))
# plt.title("20 paths, 20 ODs, pop_size = 8, gen = 400")
# plt.xlabel("time")
# plt.ylabel("fitness")
# plt.legend(['CMA', 'PSO', 'FST-PSO', 'rv_uniform', 'rv_gauss', 'vanilla'])
# plt.show()

# ALL CASES FOR ALG _
# _ = uni
# for i in range(len(_.fitness)):
#     # if _.fitness[i] != 0:
#     x = to_matrix(_.upper_time[i])[1:]
#     y = to_matrix(_.fit_update[i])
#     plt.plot(x, y, label=str(0))
# plt.title("rv_uniform\n20 paths, 20 ODs, pop_size = 8, gen = 400")
# plt.xlabel("time")
# plt.ylabel("fitness")
# plt.show()

# MEANS FOR EACH ALG
# CMA.drop([10, 16], axis=0, inplace=True)
# CMA.index = range(len(CMA.fitness))
# for _ in [CMA, PSO, FST, uni, gauss, torch]:
#     a = len(to_matrix(_.fit_update[0]))
#     b = len(_.fitness)
#     time = []
#     fit = []
#     for j in range(a):
#         x = []
#         y = []
#         for i in range(b):
#             if _ is PSO:
#                 x.append(to_matrix(_.upper_time[i])[j])
#             else:
#                 x.append(to_matrix(_.upper_time[i])[j + 1])
#             y.append(to_matrix(_.fit_update[i])[j])
#         time.append(np.mean(x))
#         fit.append(np.mean(y))
#     plt.plot(time, fit, label=str(0))
# plt.title("Mean\n20 paths, 20 ODs, pop_size = 8, gen = 400")
# plt.xlabel("time")
# plt.ylabel("fitness")
# plt.legend(['CMA', 'PSO', 'FST-PSO', 'rv_uniform', 'rv_gauss', 'vanilla'])
# plt.show()

# BOX PLOT
# plt.boxplot([CMA.fitness, PSO.fitness, FST.fitness, uni.fitness, gauss.fitness, torch.fitness])
# plt.xticks([1, 2, 3, 4, 5, 6], ['CMA', 'PSO', 'FST-PSO', 'rv_uniform', 'rv_gauss', 'vanilla'])
# plt.ylabel('fitness')
# plt.title("20 paths, 20 ODs, pop_size = 8, gen = 400")
# plt.show()

plt.boxplot([CMA.fitness, PSO.fitness, FST.fitness])
plt.xticks([1, 2, 3], ['CMA', 'PSO', 'FST-PSO'])
plt.ylabel('fitness')
plt.title("20 paths, 20 ODs, pop_size = 8, gen = 400")
plt.show()

plt.boxplot([uni.fitness, gauss.fitness, torch.fitness])
plt.xticks([1, 2, 3], ['rv_uniform', 'rv_gauss', 'vanilla'])
plt.ylabel('fitness')
plt.title("20 paths, 20 ODs, pop_size = 8, gen = 400")
plt.show()

# BEST ALG COMPARISON
# for _ in [CMA, PSO, FST]:
#     a = _.loc[_.fitness == max(_.fitness)].index
#     if _ is CMA:
#         x = to_matrix(_.upper_time[a[0]])[-350:]
#         y = to_matrix(_.fit_update[a[0]])[-350:]
#     elif _ is PSO:
#         x = to_matrix(_.upper_time[a[0]])[-320:]
#         y = to_matrix(_.fit_update[a[0]])[-320:]
#     elif _ is FST:
#         x = to_matrix(_.upper_time[a[0]])[-373:]
#         y = to_matrix(_.fit_update[a[0]])[-373:]
#     plt.plot(x, y, label=str(0))
# plt.title("20 paths, 20 ODs, pop_size = 8, gen = 400")
# plt.xlabel("time")
# plt.ylabel("fitness")
# plt.legend(['CMA', 'PSO', 'FST-PSO'])
# plt.show()

# DIFFERENT INITIAL DATA COMPARISON
# for _ in [CMA32, CMA56, CMA80]:
#     a = _.loc[_.fitness == max(_.fitness)].index
#     if _ is CMA32:
#         x = to_matrix(_.upper_time[a[0]])[:75]
#         y = to_matrix(_.fit_update[a[0]])[:75]
#     elif _ is CMA56:
#         x = to_matrix(_.upper_time[a[0]])[:45]
#         y = to_matrix(_.fit_update[a[0]])[:45]
#     elif _ is CMA80:
#         x = to_matrix(_.upper_time[a[0]])[:30]
#         y = to_matrix(_.fit_update[a[0]])[:30]
#     plt.plot(x, y, label=str(0))
# plt.title("20 paths, 20 ODs")
# plt.xlabel("time")
# plt.ylabel("fitness")
# plt.legend(['CMA32', 'CMA56', 'CMA80'])
# plt.show()
