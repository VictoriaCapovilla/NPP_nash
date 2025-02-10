import ast

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import torch


# Convert the cleaned string to an array
def to_matrix(a):
    cleaned_string = (a.replace('\r\n', ' ').replace('\n', ' ').replace('array(', '').replace('np.float64(', '')
                      .replace(', dtype=float32)', '').replace('   ', ' ').replace('  ', ' ')
                      .replace('[ ', '[').replace(' ]', ']').replace('(', '').replace(')', '')
                      .replace(' ', ',').replace(',,', ',').replace(',,', ','))

    return torch.tensor(ast.literal_eval(cleaned_string))


# # for probs_03eps_4000iter and lowest
# df = pd.read_csv(r'C:\Users\viki\Desktop\NPP\Results\Probs_study\20x20\probs_03eps_4000iter')
#
# total_paths = df.n_paths[0] + 1
# od = df.n_od[0]
#
# # update probs graph best k
# k = 10
#
# all_probs = to_matrix(df.probs_update[0])
# n_iter = to_matrix(df.lower_iter[0])
# probs = all_probs[:n_iter[0]]
# diff = abs(probs[-1] - probs[0])
# # print(diff)
#
# a = torch.topk(diff.flatten(), k=k).indices
# print(a)
#
# indexes = []
# for _ in a:
#     print(_)
#     row = int(_ / total_paths)
#     col = int(_ - total_paths * row)
#     val = diff[row, col]
#     print('val =', val, '\nindexes =', [row, col])
#     print('initial value = ', probs[0][row, col], '\nlast value = ', probs[-1][row, col])
#
#     indexes.append([row, col])
#
# print(indexes)
#
# for ind in indexes:
#     x = [_ for _ in range(n_iter[0])]
#     y = [np.array(probs[i][ind[0], ind[1]]) for i in range(n_iter[0])]
#     plt.plot(x, y, label=str(0))
# plt.title("Best " + str(k) + " probs update in one generation\n" + str(total_paths - 1) + "x" + str(od) + "_" +
#           str(df.pop_size[0]) + ", lower_eps=" + str(df.eps[0]) + ", lower_iter=" + str(int(n_iter[0])))
# plt.xlabel("Iteration")
# plt.ylabel("Probability")
# plt.show()


# for probs_04eps_5000iter and greatest
df = pd.read_csv(r'C:\Users\viki\Desktop\NPP\Results\Probs_study\90x90\probs_04eps_5000iter')

total_paths = df.n_paths[0] + 1
od = df.n_od[0]

# update probs graph best k
k = 10

n_iter = to_matrix(df.lower_iter[0])
probs = to_matrix(df.probs_update[0])
for i in range(k):
    x = [_ for _ in range(n_iter[0])]
    y = np.array(probs[i])
    plt.plot(x, y, label=str(0))
plt.title("Best " + str(k) + " probs update in one generation\n" + str(total_paths - 1) + "x" + str(od) + "_" +
          str(df.pop_size[0]) + ", lower_eps=" + str(df.eps[0]) + ", lower_iter=" + str(int(n_iter[0])))
plt.xlabel("Iteration")
plt.ylabel("Probability")
plt.show()
