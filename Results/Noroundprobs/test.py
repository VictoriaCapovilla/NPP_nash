import ast

import numpy as np
import pandas as pd


# transformation from strings to arrays
def to_matrix(a):
    cleaned_string = (a.replace('\r\n', ' ').replace('\n', ' ').strip().replace('   ',' ').replace('  ',' ')
                      .replace('[ ','[').replace(' ]',']').replace(' ',',').replace(',,',','))

    # Step 2: Convert the cleaned string to a list of lists
    return np.array(ast.literal_eval(cleaned_string))

# to transformation from tensor to float
def to_float(a):
    cleaned_string = (a.replace('tensor(', '').replace(", device='cuda:0', dtype=torch.float64)", '').strip().replace('"',''))

    return np.array(ast.literal_eval(cleaned_string))


# same instance
df4 = pd.read_csv('/Results/Noroundprobs/output_eps-4')
df5 = pd.read_csv('/Results/Noroundprobs/output_eps-5')
df6 = pd.read_csv('/Results/Noroundprobs/output_eps-6')

# df4s = pd.read_csv('C:/Users/viki/Desktop/NPP/Results/Noroundprobs/study_eps-4')
# df5s = pd.read_csv('C:/Users/viki/Desktop/NPP/Results/Noroundprobs/study_eps-5')
# df6s = pd.read_csv('C:/Users/viki/Desktop/NPP/Results/Noroundprobs/study_eps-6')

# rounded probabilities
df_rprob = pd.read_csv('/Results/output_roundprobs')
df_rprobs = pd.read_csv('/Results/study_roundprobs')

# for _ in [df0, df1, df2, df3, df4, df5, df6]:
#     _.best_individual = _.best_individual.apply(to_matrix)
#     _.probs = _.probs.apply(to_matrix)
#     _.payoffs = _.payoffs.apply(to_matrix)

# df1.probs.iloc[0]
# print(dfprova.probs.iloc[0])

# dfprova.rename({'best individual': 'best_individual'}, axis=1, inplace=True)
# dfprova.to_csv(r"Results/prova", index=False)

# df.fitness = df.fitness.apply(to_float)

df_study = None
for _ in [df]:
    _.best_individual = _.best_individual.apply(to_matrix)
    _.probs = _.probs.apply(to_matrix)
    _.payoffs = _.payoffs.apply(to_matrix)

    print('\ndataframe information:')
    df = _
    x = 0
    a = 0
    b = 0
    c = 0
    for i in range(0, len(df.fitness)-1):
        # if the next fitness value is smaller than the actual fitness value, in the same run
        # if df.fitness[i] - df.fitness[i + 1] > 10**(-3) and df.run[i + 1] == df.run[i]:
        if df.fitness[i] > df.fitness[i + 1] and df.run[i + 1] == df.run[i]:
            if df.n_iter[i] < df.n_iter[i + 1]:
                x += 1
                # print('\nsmall fitness, big iter =\n',
                #       pd.concat([df.fitness[i:i + 2], df.n_iter[i:i + 2], df.eps[i:i + 2], df.run[i:i + 2]],axis=1))
                # print('fitness difference = ', (df.fitness[i] - df.fitness[i + 1]))
                # print('the individuals are equal:', (df.best_individual[i] == df.best_individual[i + 1]).all())
                #
                # print('the probabilities are the same:', (df.probs[i] == df.probs[i + 1]).all())
                # print('probs difference, min = ', (df.probs[i] - df.probs[i + 1]).min())
                # print('probs difference, max = ', (df.probs[i] - df.probs[i + 1]).max())
                # print('probs difference, mean = ', (df.probs[i] - df.probs[i + 1]).mean())
                #
                # print('the payoffs are the same:', (df.payoffs[i] == df.payoffs[i + 1]).all())
                # print('payoff difference, min = ', (df.payoffs[i] - df.payoffs[i + 1]).min())
                # print('payoff difference, max = ', (df.payoffs[i] - df.payoffs[i + 1]).max())
                # print('payoff difference, mean = ', (df.payoffs[i] - df.payoffs[i + 1]).mean())
            # if the next number of iterations is smaller than the actual number of iterations
            elif df.n_iter[i] > df.n_iter[i + 1] and (df.best_individual[i] == df.best_individual[i + 1]).all():
                if not (df.payoffs[i] == df.payoffs[i + 1]).all():
                    a += 1
                    # print('\nsmall fitness, small iter, same ind, different payoffs =\n',
                    #       pd.concat([df.fitness[i:i + 2], df.n_iter[i:i + 2], df.eps[i:i + 2], df.run[i:i + 2]], axis=1))
                    # print('fitness difference = ', (df.fitness[i] - df.fitness[i + 1]))
                    # print('payoff difference, min = ', (df.payoffs[i] - df.payoffs[i + 1]).min())
                    # print('payoff difference, max = ', (df.payoffs[i] - df.payoffs[i + 1]).max())
                    # print('payoff difference, mean = ', (df.payoffs[i] - df.payoffs[i + 1]).mean())
                elif (df.payoffs[i] == df.payoffs[i + 1]).all():
                    b += 1
                    # print('\nsmall fitness, small iter, same ind, same payoffs =\n',
                    #       pd.concat([df.fitness[i:i + 2], df.n_iter[i:i + 2], df.eps[i:i + 2], df.run[i:i + 2]],axis=1))
                    # print('fitness difference = ', (df.fitness[i] - df.fitness[i + 1]))
                    # print('the probabilities are the same:', (df.probs[i] == df.probs[i + 1]).all())
                    # print('probs difference, min = ', (df.probs[i] - df.probs[i + 1]).min())
                    # print('probs difference, max = ', (df.probs[i] - df.probs[i + 1]).max())
                    # print('probs difference, mean = ', (df.probs[i] - df.probs[i + 1]).mean())
            elif (df.n_iter[i] > df.n_iter[i + 1] and
                  not (df.best_individual[i] == df.best_individual[i + 1]).all()):
                c += 1

            # update dataframe
            if df_study is None:
                df_study = df[i:i + 2]
            else:
                df_study = pd.concat([df_study, df[i:i + 2]])

    print('\ntotal cases =', len(df.fitness))
    print('small fitness, big iter =', x)
    print('small fitness, small iter, same ind, different payoffs =', a)
    print('small fitness, small iter, same ind, same payoffs =', b)
    print('small fitness, small iter, different ind =', c)

df_study.to_csv('C:/Users/viki/Desktop/NPP/Results/study_output', index=False)
df_output = pd.read_csv('C:/Users/viki/Desktop/NPP/Results/study_output')

for _ in [df_output]:
    _.best_individual = _.best_individual.apply(to_matrix)
    _.probs = _.probs.apply(to_matrix)
    _.payoffs = _.payoffs.apply(to_matrix)

    iter = []
    fit = []
    diff_fit = []
    ind = []
    same_ind = []
    probs = []
    min_diff_probs = []
    max_diff_probs = []
    mean_diff_probs = []
    payoffs = []
    min_diff_payoffs = []
    max_diff_payoffs = []
    mean_diff_payoffs = []
    run = []
    i = 0
    while i < len(_.fitness):
        iter.append(np.array([_.n_iter[i], _.n_iter[i + 1]]))
        fit.append(np.array([_.fitness[i], _.fitness[i + 1]]))
        diff_fit.append(_.fitness[i] - _.fitness[i + 1])
        ind.append(np.array([_.best_individual[i], _.best_individual[i + 1]]))
        same_ind.append((_.best_individual[i] == _.best_individual[i + 1]).all())
        probs.append(np.array([_.probs[i], _.probs[i + 1]]))
        min_diff_probs.append((_.probs[i] - _.probs[i + 1]).min())
        max_diff_probs.append((_.probs[i] - _.probs[i + 1]).max())
        mean_diff_probs.append((_.probs[i] - _.probs[i + 1]).mean())
        payoffs.append(np.array([_.payoffs[i], _.payoffs[i + 1]]))
        min_diff_payoffs.append((_.payoffs[i] - _.payoffs[i + 1]).min())
        max_diff_payoffs.append((_.payoffs[i] - _.payoffs[i + 1]).max())
        mean_diff_payoffs.append((_.payoffs[i] - _.payoffs[i + 1]).mean())
        run.append(_.run[i])
        i += 2

    data = {
        'iter': iter,
        'fit': fit,
        'diff_fit': diff_fit,
        'ind': ind,
        'same_ind': same_ind,
        'probs': probs,
        'min_diff_probs': min_diff_probs,
        'max_diff_probs': max_diff_probs,
        'mean_diff_probs': mean_diff_probs,
        'payoffs': payoffs,
        'min_diff_payoffs': min_diff_payoffs,
        'max_diff_payoffs': max_diff_payoffs,
        'mean_diff_payoffs': mean_diff_payoffs,
        'eps': [_.eps[0] for k in range(len(iter))],
        'run': run
    }

    x = pd.DataFrame(data=data)
    x.to_csv('C:/Users/viki/Desktop/NPP/Results/study_output', index=False)
    df_output = pd.read_csv('C:/Users/viki/Desktop/NPP/Results/study_output')
