import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('results.csv', sep=';')

accuracy = df.groupby(['aggregator'])['accuracy']

plt.Figure()

plt.title('MUTAG Classification Accuracy\n10-fold cross-validation')

aggregators = df['aggregator'].unique()

data = [df.loc[df['aggregator'] == agg]['accuracy'] for agg in aggregators]

plt.boxplot(data)

plt.xticks(ticks=np.arange(1, len(aggregators) + 1), labels=aggregators)

plt.show()
