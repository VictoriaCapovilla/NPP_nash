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


df128_10 = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/Par/a04b08/128_10')
df128_50 = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/Par/a04b08/128_50')
df128_100 = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/Par/a04b08/128_100')
df256_10 = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/Par/a04b08/256_10')
df256_50 = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/Par/a04b08/256_50')
df256_100 = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/Par/a04b08/256_100')
df512_10 = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/Par/a04b08/512_10')
df512_50 = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/Par/a04b08/512_50')
df512_100 = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/Par/a04b08/512_100')

seen = set()

# # SAME POP SIZE _ DIFFERENT GENERATIONS
cases128 = [x for x in df128_10['case'] if not (x in seen or seen.add(x))]
cases256 = [x for x in df256_10['case'] if not (x in seen or seen.add(x))]
cases512 = [x for x in df512_10['case'] if not (x in seen or seen.add(x))]
cases_pop = [cases128, cases256, cases512]
dfs_pop = [[df128_100, df128_50, df128_10],[df256_100, df256_50, df256_10],[df512_100, df512_50, df512_10]]
colors = ['darkorange', 'lime', 'm']
# MEANS FOR EACH ALG
for (p, k) in zip(dfs_pop,cases_pop):
    for case in k:
        for t, _ in enumerate(p):
            df = _.loc[_['case'] == case]
            a = len(to_matrix(df.fit_update.iloc[0]))
            b = len(df.fitness)
            time = []
            fit = []
            for j in range(a):
                x = []
                y = []
                x.append(j+1)
                for i in range(b):
                    y.append(to_matrix(df.fit_update.iloc[i])[j])
                time.append(np.mean(x))
                fit.append(np.mean(y))
            plt.plot(time, fit, label=str(0), color = colors[t])
        plt.title("Mean\n" + str(df.n_paths.iloc[0]) + " paths, " + str(df.n_od.iloc[0]) + " ODs, pop_size = " + str(df.pop_size.iloc[0]))
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend([str(p[0].upper_iter.iloc[0]),str(p[1].upper_iter.iloc[0]),str(p[2].upper_iter.iloc[0])], title='total gen')
        plt.savefig('mean' + str(df.n_paths.iloc[0]) + 'x' + str(df.n_od.iloc[0]) + '_pop' + str(df.pop_size.iloc[0]))
        # plt.show()
        plt.cla()
        plt.clf()

# BOX PLOTS
dfs_pop = [[df128_10, df128_50, df128_100],[df256_10, df256_50, df256_100],[df512_10, df512_50, df512_100]]
for (p, t) in zip(dfs_pop,cases_pop):
    for case in t:
        df0 = p[0].loc[p[0]['case'] == case]
        df1 = p[1].loc[p[1]['case'] == case]
        df2 = p[2].loc[p[2]['case'] == case]
        plt.boxplot([df0.fitness,df1.fitness,df2.fitness])
        plt.xticks([1, 2, 3], [str(df0.upper_iter.iloc[0]),
                               str(df1.upper_iter.iloc[0]),
                               str(df2.upper_iter.iloc[0])])
        plt.ylabel('fitness')
        plt.title("Meanbox\n" + str(df0.n_paths.iloc[0]) +
                  " paths, " + str(df0.n_od.iloc[0]) +
                  " ODs, pop_size = " + str(df0.pop_size.iloc[0]))
        plt.savefig('meanbox' + str(df0.n_paths.iloc[0]) + 'x' + str(df0.n_od.iloc[0]) + '_' + str(df0.pop_size.iloc[0]))
        # plt.show()
        plt.cla()
        plt.clf()

# # DIFFERENT POP SIZE _ SAME GENERATIONS
cases_gen = [[cases512[i],cases256[i],cases128[i]] for i in range(len(cases128))]
dfs_gen = [[df512_10, df256_10, df128_10],[df512_50, df256_50, df128_50],[df512_100, df256_100, df128_100]]
# MEANS FOR EACH ALG
for p in dfs_gen:
    for k in cases_gen:
        for t, (_,case) in enumerate(zip(p,k)):
            df = _.loc[_['case'] == case]
            a = len(to_matrix(df.fit_update.iloc[0]))
            b = len(df.fitness)
            time = []
            fit = []
            for j in range(a):
                x = []
                y = []
                x.append(j + 1)
                for i in range(b):
                    y.append(to_matrix(df.fit_update.iloc[i])[j])
                time.append(np.mean(x))
                fit.append(np.mean(y))
            plt.plot(time, fit, label=str(0), color = colors[t])
        plt.title("Mean\n" + str(df.n_paths.iloc[0]) + " paths, " + str(df.n_od.iloc[0]) + " ODs" +
                  ", gen = " + str(df.upper_iter.iloc[0]))
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend([str(p[0].pop_size.iloc[0]),str(p[1].pop_size.iloc[0]),str(p[2].pop_size.iloc[0])])
        plt.savefig('mean' + str(df.n_paths.iloc[0]) + 'x' + str(df.n_od.iloc[0]) + '_gen' + str(df.upper_iter.iloc[0]))
        # plt.show()
        plt.cla()
        plt.clf()

# BOX PLOTS
for p in dfs_gen:
    for case in cases_gen:
        df0 = p[0].loc[p[0]['case'] == case[0]]
        df1 = p[1].loc[p[1]['case'] == case[1]]
        df2 = p[2].loc[p[2]['case'] == case[2]]
        plt.boxplot([df0.fitness,df1.fitness,df2.fitness])
        plt.xticks([1, 2, 3], [str(df0.pop_size.iloc[0]),
                               str(df1.pop_size.iloc[0]),
                               str(df2.pop_size.iloc[0])])
        plt.ylabel('fitness')
        plt.title("Mean\n" + str(df0.n_paths.iloc[0]) +
                  " paths, " + str(df0.n_od.iloc[0]) +
                  " ODs, gen = " + str(df0.upper_iter.iloc[0]))
        plt.savefig('meanbox' + str(df0.n_paths.iloc[0]) + 'x' + str(df0.n_od.iloc[0]) + '_gen' + str(df0.upper_iter.iloc[0]))
        # plt.show()
        plt.cla()
        plt.clf()
