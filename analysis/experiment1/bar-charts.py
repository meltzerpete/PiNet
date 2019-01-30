import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# prepare data
df = pd.read_csv("results-2019-01-29.csv", sep=";")
df.drop(df.loc[df['normalise_by_num_nodes']].index, inplace=True)
ptc = df.loc[df['dataset'].str.startswith("PTC")] \
    .groupby(["preprocessA", "normalise_by_num_nodes"], as_index=False).mean()
ptc['dataset'] = 'PTC'
ptc['pretty_name'] = 'PTC'

df.drop(df.loc[df['dataset'].str.startswith("PTC")].index, inplace=True)
df = df.append(ptc, sort=False)

# plot by dataset

datasets = df['dataset'].values
for dataset in set(datasets):
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    df_dataset_dataset_ = df.loc[df['dataset'] == dataset]

    plt.title(df_dataset_dataset_['pretty_name'].any())

    xticks = np.arange(len(df_dataset_dataset_))
    plt.bar(xticks, df_dataset_dataset_['mean_acc'])
    plt.xticks(xticks, map(lambda row: row[0] + ("norm'd" if row[1] else ""),
               df_dataset_dataset_[['preprocessA', 'normalise_by_num_nodes']].values),
               rotation=90)
    plt.show()
