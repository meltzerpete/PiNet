import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# prepare data
df = pd.read_csv("results-2019-01-29.csv", sep=";")
df.drop(df.loc[df['normalise_by_num_nodes']].index, inplace=True)
ptc = df.loc[df['dataset'].str.startswith("PTC")] \
    .groupby(["preprocessA", "normalise_by_num_nodes"], as_index=False).mean()
ptc['dataset'] = 'PTC'
ptc['pretty_name'] = 'PTC'

df.drop(df.loc[df['dataset'].str.startswith("PTC")].index, inplace=True)
df = df.append(ptc, sort=False)

df.sort_values(['dataset', 'preprocessA'], inplace=True)

df = df.loc[df['dataset'] != 'ENZYMES']
df.drop(df[df['preprocessA'].str.contains('laplacian')].index, inplace=True)

fig, axes = plt.subplots(3, 2, sharex='col', sharey='none')
plt_positions = [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)]
y_lims = [(.825, .895), (.7, .75), (.68, .74), (.73, .76), (.605, .635)]

for i, dataset, ylim in zip(plt_positions, df['dataset'].unique(), y_lims):
    print(i, dataset, ylim)
    ax = axes[i]

    data = df.loc[df['dataset'] == dataset]
    title = data['pretty_name'].any()
    ax.set_title(title)

    xticks = np.arange(len(data))
    for x, y in zip(xticks, data['mean_acc']):
        ax.bar(x, y)
    ax.set_xticks([])
    ax.minorticks_on()
    ax.set_ylim(ylim)

    # mean classification of manual search space
    manData = data.drop(data[data['preprocessA'].str.contains('PQ')].index)
    manMean = manData['mean_acc'].mean()
    ax.plot([-.5, len(manData) + 0.5], [manMean] * 2, '--', color='black', alpha=.5)


# legend
lines = [Line2D([0], [0], color=c) for c in ['blue', 'orange', 'green', 'red', 'purple', 'brown']]
axes[0, 1].axis('off')
axes[0, 1].legend(lines, ['Sym_norm(A + I)', 'A + I', 'Sym_norm(A)', 'A', 'Learn $p$ and $q$'], loc='center')

fig.suptitle('Message Passing Matrix vs. Mean Classification Accuracy')
fig.tight_layout(rect=[0, -0.02, 1, 0.95])
# fig.set_figheight(.9 * fig.get_figheight())
fig.savefig('all-matrices.pdf')
fig.show()

df = df.reset_index()

for dataset in df['dataset'].unique():
    d = df.loc[df['dataset'] == dataset]
    idxmax = d['mean_acc'].idxmax()
    print(idxmax)
    matrix = df['preprocessA'].iloc[idxmax]
    print(dataset, matrix, f'{df["mean_acc"].iloc[idxmax]:.2f}', '+/-', f'{df["acc_std"].iloc[idxmax]:.2f}')
