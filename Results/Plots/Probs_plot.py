import ast

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt


# Convert the cleaned string to an array
def to_matrix(a):
    cleaned_string = (a.replace('\r\n', ' ').replace('\n', ' ').replace('array(', '').replace('np.float64(', '')
                      .replace(', dtype=float32)', '').replace('   ', ' ').replace('  ', ' ')
                      .replace('[ ', '[').replace(' ]', ']').replace('(', '').replace(')', '')
                      .replace(' ', ',').replace(',,', ',').replace(',,', ','))

    return torch.tensor(ast.literal_eval(cleaned_string))
    # return np.array(ast.literal_eval(cleaned_string), dtype=object)


df = pd.read_csv(r'C:\Users\viki\Desktop\NPP\Results\Probs_study\90x90\instance5.txt')
all_probs = np.load(r'C:\Users\viki\Desktop\NPP\Results\Probs_study\90x90\test5.npy')

total_paths = df.n_paths[0] + 1
od = df.n_od[0]

n_iter = to_matrix(df.lower_iter[0])
x = [_ for _ in range(n_iter[-1])]
for j in range(all_probs.shape[2]):
    print(j)
    for i in range(all_probs.shape[1]):
        y = [np.array(all_probs[k][i, j]) for k in range(all_probs.shape[0])]
        plt.plot(x, y, label=str(0))
    print('Plotting...')
    plt.title("All probs update in one generation\n" + str(total_paths - 1) + "x" + str(od) + "_" +
              str(df.pop_size[0]) + ", lower_eps=" + str(df.eps[0]) + ", lower_iter=" + str(int(n_iter[-1])))
    plt.xlabel("Iteration")
    plt.ylabel("Probability")
    plt.show()
