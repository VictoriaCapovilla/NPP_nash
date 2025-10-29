import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True})

df = pd.read_csv('Results/random_comparison.csv')

df['case'] = df.n_od.astype(str) + ' ' + df.paths.astype(str)

df['ga_best'] = df.best_val >= df.rand_best_val
df['gap'] = (df.best_val - df.rand_best_val) * 100 / df.rand_best_val

'''RANDOM COMPARISON'''

df_res = df.groupby('case').agg(
    {'n_od': ['max'], 'paths': ['max'], 'ga_best': ['sum'], 'gap': ['mean', 'std'], 'time': ['mean', 'std'],
     'rand_time': ['mean', 'std']})


def format_dataframe_for_latex(df, float_columns):
    # Crea una copia per non modificare l'originale
    df_formatted = df.copy()

    # Lista delle colonne da formattare
    cols_to_format = []
    for col in df.columns:
        col_name = col[0] if isinstance(col, tuple) else col
        if any(keyword in str(col_name) for keyword in float_columns):
            cols_to_format.append(col)

    # Applica la formattazione
    for col in cols_to_format:
        df_formatted[col] = df_formatted[col].apply(
            lambda x: f'{float(x):.2f}' if pd.notnull(x) and str(x).replace('.', '').isdigit() else str(x)
        )

    return df_formatted


# Formatta il DataFrame
df_formatted = format_dataframe_for_latex(df_res, ['gap', 'time', 'rand_time'])

# Genera il codice LaTeX
latex_code = df_formatted.to_latex(escape=False, multicolumn=True, multicolumn_format='c', multirow=True, index=False,
                                   caption="Performance Comparison: GA vs. RS",
                                   label="tab:1", position='h!')

print(latex_code)

df_formatted = format_dataframe_for_latex(df_res[df_res.columns[:5]], ['gap'])

latex_code = df_formatted.to_latex(escape=False, multicolumn=True, multicolumn_format='c', multirow=True, index=False,
                                   caption="Performance Comparison: GA vs. RS",
                                   label="tab:1", position='h!')

print(latex_code)







'''LOWER COMPARISON'''

df_l = pd.read_csv('Results/random_lower_comparison.csv')

min_iter = df_l.iterations.min()
g_times, r_times = np.zeros((df_l.shape[0], min_iter)), np.zeros((df_l.shape[0], min_iter))
g_iter, r_iter = np.zeros((df_l.shape[0], min_iter)), np.zeros((df_l.shape[0], min_iter))
g_vals, r_vals = np.zeros((df_l.shape[0], min_iter)), np.zeros((df_l.shape[0], min_iter))
g_times_norm, r_times_norm = np.zeros((df_l.shape[0], min_iter)), np.zeros((df_l.shape[0], min_iter))
for i, row in df_l.iterrows():
    g_times[i] = eval(row.ga_l_times)[:min_iter]
    r_times[i] = eval(row.rs_l_times)[:min_iter]
    g_iter[i] = eval(row.ga_l_iter)[:min_iter]
    r_iter[i] = eval(row.rs_l_iter)[:min_iter]
    g_vals[i] = eval(row.ga_vals)[:min_iter]
    g_vals[i] = g_vals[i] / g_vals[i].max()
    r_vals[i] = eval(row.rs_vals)[:min_iter]
    g_times_norm[i] = g_times[i] / g_times[i].max()

print(np.corrcoef(g_times.flatten(), g_iter.flatten()))

g_mean = g_times.mean(axis=0)
g_std = g_times.std(axis=0)

r_mean = r_times.mean(axis=0)
r_std = r_times.std(axis=0)

fig_size = (24, 18)
dpi = 150
plt.figure(figsize=fig_size, dpi=dpi)
colors = ['#1f77b4', '#ff7f0e']

# Imposta il fontsize gobale
plt.rcParams.update({'font.size': 30})

plt.plot(range(min_iter), g_mean, label='GA', color=colors[0], linewidth=5, marker='o', markersize=20)
plt.fill_between(range(min_iter), g_mean - g_std, g_mean + g_std, alpha=0.2, color=colors[0])
plt.plot(range(min_iter), r_mean, label='RS', color=colors[1], linewidth=5, marker='o', markersize=20)
plt.fill_between(range(min_iter), r_mean - r_std, r_mean + r_std, alpha=0.2, color=colors[1])

# Personalizza il plot con fontsize 50
plt.xlabel('GA iter', fontsize=50)
plt.ylabel('Iter time (seconds)', fontsize=50)
plt.legend(fontsize=50)
plt.grid(True, alpha=0.3)
# Aumenta la dimensione dei tick labels
plt.tick_params(axis='both', which='major', labelsize=50)

plt.tight_layout()
plt.show()

plt.figure(figsize=fig_size, dpi=dpi)
plt.scatter(g_vals.flatten(), g_times_norm.flatten(), s=100)
plt.xlabel('Fitness / Max Fitness', fontsize=50)
plt.ylabel(r'$\mathcal{N} Iter time / Max $\mathcal{N} Iter time$', fontsize=50)
plt.tick_params(axis='both', which='major', labelsize=50)
plt.show()






'''SCALABITLITY '''

df_scalability = df_res[[('n_od', 'max'), ('paths', 'max'), ('time', 'mean')]]
df_scalability.columns = ['n_od', 'paths', 'time']
df_std = df_res[[('n_od', 'max'), ('paths', 'max'), ('time', 'std')]]
df_std.columns = ['n_od', 'paths', 'time_std']

# Pivot per avere ODs come indice e Paths come colonne
pivot_table = df_scalability.pivot(index='n_od', columns='paths', values='time')
pivot_table_std = df_std.pivot(index='n_od', columns='paths', values='time_std')

latex_code1 = pivot_table.to_latex(float_format="%.2f", caption="Mean GA Time (sec)", label="tab:time_matrix", position='h!')
print(latex_code1)
latex_code2 = pivot_table_std.to_latex(float_format="%.2f", caption="STD GA Time (sec)", label="tab:time_matrix_std", position='h!')
print(latex_code2)



fig_size = (24, 18)
dpi = 300
plt.figure(figsize=fig_size, dpi=dpi)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Imposta il fontsize gobale
plt.rcParams.update({'font.size': 30})

# Crea il plot per ogni numero di ODs
for i, (od_value, color) in enumerate(zip(pivot_table.index, colors)):
    means = pivot_table.loc[od_value].to_numpy()
    stds = pivot_table_std.loc[od_value].to_numpy()
    paths = pivot_table.columns.to_numpy()

    # Plot della linea principale
    plt.plot(paths, means, label=f'{od_value} ODs', color=color, linewidth=5, marker='o', markersize=20)

    # Area della deviazione standard
    plt.fill_between(paths, means - stds, means + stds, alpha=0.2, color=color)

# Personalizza il plot con fontsize 50
plt.xlabel('Number of Paths', fontsize=50)
plt.ylabel('GA Time (seconds)', fontsize=50)
plt.legend(fontsize=50)
plt.grid(True, alpha=0.3)
plt.xticks([10, 20, 30])
# Aumenta la dimensione dei tick labels
plt.tick_params(axis='both', which='major', labelsize=50)

plt.tight_layout()
plt.show()

plt.figure(figsize=fig_size, dpi=dpi)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Colori per i diversi Paths

# Imposta il fontsize 50obale
plt.rcParams.update({'font.size': 30})

# Crea il plot per ogni numero di Paths
for i, (path_value, color) in enumerate(zip(pivot_table.index, colors)):
    means = pivot_table[path_value].to_numpy()
    stds = pivot_table_std[path_value].to_numpy()
    ods = pivot_table.columns.to_numpy()  # Ora ODs sono le colonne

    # Plot della linea principale
    plt.plot(ods, means, label=f'{path_value} Paths', color=color, linewidth=5, marker='o', markersize=20)

    # Area della deviazione standard
    plt.fill_between(ods, means - stds, means + stds, alpha=0.2, color=color)

# Personalizza il plot con fontsize 50
plt.xlabel('Number of ODs', fontsize=50)
plt.ylabel('GA Time (seconds)', fontsize=50)
plt.legend(fontsize=50)
plt.grid(True, alpha=0.3)
plt.xticks([10, 20, 30])

# Aumenta la dimensione dei tick labels
plt.tick_params(axis='both', which='major', labelsize=50)

plt.tight_layout()
plt.show()

df['lower_per_iter'] = df.time / df.iterations

df_lower = df.groupby('case').agg({'iterations': 'mean', 'time': 'mean', 'lower_per_iter': 'mean'})

df_low_latex = format_dataframe_for_latex(df_lower, df_lower.columns)

latex_code = df_low_latex.to_latex(
    escape=False,
    multirow=True,
    index=True,  # o False se non vuoi l'indice
    caption="GA iterations and N(T) time",
    label="tab:lower",
    position='h!'  # posizionamento in LaTeX
)

print(latex_code)











''' PARAMS ANALYSIS '''

df_params = pd.read_csv('Results/params_comparison.csv')
best_val = {}
for run in df_params.run.unique():
    best_val[run] = df_params[df_params.run == run].best_val.max()

df_params['best_all'] = df_params.run.apply(lambda x: best_val[x])
df_params['is_best'] = df_params.best_val == df_params.best_all
df_params['gap'] = (1 - df_params.best_val/df_params.best_all) * 100

df_p = df_params.groupby('pop_size').agg({'is_best': 'sum', 'gap':['mean', 'std'], 'time': ['mean', 'std'], 'iterations': ['mean', 'std']})
df_p_latex = format_dataframe_for_latex(df_p, ['gap', 'time', 'iterations'])

latex_code = df_p_latex.to_latex(
    escape=False,
    multicolumn=True,
    multicolumn_format='c',
    multirow=True,
    index=True,  # o False se non vuoi l'indice
    caption="Population size analysis",
    label="tab:pop_size",
    position='h!'  # posizionamento in LaTeX
)
print(latex_code)
