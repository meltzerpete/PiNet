import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# prepare data
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

df = pd.read_csv("results-2019-01-29.csv", sep=";")
df.drop(df.loc[df['normalise_by_num_nodes']].index, inplace=True)
ptc = df.loc[df['dataset'].str.startswith("PTC")] \
    .groupby(["preprocessA", "normalise_by_num_nodes"], as_index=False).mean()
ptc['dataset'] = 'PTC'
ptc['pretty_name'] = 'PTC'

df.drop(df.loc[df['dataset'].str.startswith("PTC")].index, inplace=True)
df = df.append(ptc, sort=False)

df.sort_values(['dataset', 'preprocessA'], inplace=True)

# plot by dataset

# datasets = df['dataset'].values
# for dataset in set(datasets):
#     fig = plt.Figure()
#     ax = fig.add_subplot(111)
#     df_dataset_dataset_ = df.loc[df['dataset'] == dataset]
#
#     title = df_dataset_dataset_['pretty_name'].any()
#     plt.title(title)
#
#     xticks = np.arange(len(df_dataset_dataset_))
#     plt.bar(xticks, df_dataset_dataset_['mean_acc'])
#     plt.xticks(xticks, map(lambda row: row[0] + ("norm'd" if row[1] else ""),
#                df_dataset_dataset_[['preprocessA', 'normalise_by_num_nodes']].values),
#                rotation=90)
#     plt.yscale('log')
#     plt.tight_layout()
#     plt.savefig(f'{title}-matrices.pdf')
#     plt.show()

df = df.loc[df['dataset'] != 'ENZYMES']

fig, axes = plt.subplots(3, 2, sharex='col', sharey='none', tight_layout=True)
plt_positions = [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)]
y_lims = [(.675, .88), (.7, .75), (.68, .74), (.72, .75), (.605, .635)]

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
    # ax.set_yscale('log')
    # ax.set_ylim(data['mean_acc'].min() * 0.99, data['mean_acc'].max() * 1.01)
    ax.set_ylim(ylim)

# legend
lines = [Line2D([0], [0], color=c) for c in ['blue', 'orange', 'green', 'red', 'purple', 'brown']]
axes[0, 1].axis('off')
axes[0, 1].legend(lines, ['Sym_norm(A + I)', 'A + I', 'L', 'Sym_norm(L)', 'Sym_norm(A)', 'A'], loc='center')

fig.tight_layout()
fig.savefig('all-matrices.pdf')
fig.show()
