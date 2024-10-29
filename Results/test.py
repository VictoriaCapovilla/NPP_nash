import numpy as np
import pandas as pd

df1 = pd.read_csv('Results/output_eps-4')
df2 = pd.read_csv('Results/output_eps-5')
df3 = pd.read_csv('Results/output_eps-6')

df1.insert(14, "eps", [10**(-4) for _ in range(606)], True)

df_study = None

for _ in [df1, df2, df3]:
    print('\ndataframe information:\n')
    df = _
    x = 0
    a = 0
    b = 0
    c = 0
    for i in range(0,len(df['fitness'])-1):
        # if the next fitness value is smaller than the actual fitness value, in the same run
        if df['fitness'][i] > df['fitness'][i + 1] and df['run'][i + 1] == df['run'][i]:
            if df['n_iter'][i] < df['n_iter'][i + 1]:
                x += 1
                # print('small fitness, big iter =\n', pd.concat([df['fitness'][i:i+2],df['n_iter'][i:i+2],df['run'][i:i+2]],axis=1))
                # print('the individuals are equal:', df['best individual'][i] == df['best individual'][i+1])
                # print('the probabilities are the same:', df['probs'][i] == df['probs'][i+1])
                # print('the payoffs are the same:', df['payoffs'][i] == df['payoffs'][i+1])
            # if the next number of iterations is smaller than the actual number of iterations
            elif df['n_iter'][i] > df['n_iter'][i + 1] and df['best individual'][i] == df['best individual'][i + 1]:
                if df_study is None:
                    df_study = df[i:i+2]
                else:
                    df_study = pd.concat([df_study, df[i:i+2]])
                if not df['payoffs'][i] == df['payoffs'][i + 1]:
                    a += 1
                    # print('small fitness, small iter, same ind, different payoffs =\n',
                    #       pd.concat([df['fitness'][i:i + 2], df['n_iter'][i:i + 2], df['run'][i:i + 2]], axis=1))
                elif df['payoffs'][i] == df['payoffs'][i + 1]:
                    b += 1
                    # print('small fitness, small iter, same ind, same payoffs =\n',
                    #       pd.concat([df['fitness'][i:i+2],df['n_iter'][i:i+2],df['run'][i:i+2]],axis=1))
                    # print('the probabilities are the same:', df['probs'][i] == df['probs'][i + 1])
            elif df['n_iter'][i] > df['n_iter'][i + 1] and not df['best individual'][i] == df['best individual'][i + 1]:
                c += 1

    df_study.to_csv(r"Results/study_diffind", index=False)

    print('\ntotal cases =', len(df['fitness']))
    print('small fitness, big iter =', x)
    print('small fitness, small iter, same ind, different payoffs =', a)
    print('small fitness, small iter, same ind, same payoffs =', b)
    print('small fitness, small iter, different ind =', c)

# df['fitness'].to_csv("Results/fitness", index=False)
# df['best individual'].to_csv("Results/best_individual", index=False)