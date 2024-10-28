import pandas as pd

df = pd.read_csv('/home/capovilla/Scrivania/NPP_nash/Results/output')

x = 0
for i in range(0,len(df['fitness'])-1):
    if df['fitness'][i+1]<df['fitness'][i]:
        if df['n_iter'][i+1]>df['n_iter'][i] and df['run'][i+1]==df['run'][i]:
            print(pd.concat([df['fitness'][i:i+2],df['n_iter'][i:i+2], df['best individual'][i:i+2]],axis=1))
        elif df['n_iter'][i+1]<df['n_iter'][i] and df['run'][i+1]==df['run'][i]:
            x += 1
print('x=',x)