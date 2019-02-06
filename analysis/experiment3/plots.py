import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('result1.csv', sep=';')

SOA = {
    'MUTAG': .889,
    'NCI1': .827,
    'NCI109': .824,
    'ENZYMES': .65,
    'PROTEINS': .764,
    'PTC': .69
}

ptc = df.loc[df['dataset'].str.startswith("PTC")]
ptc['dataset'] = 'PTC'
ptc['pretty_name'] = 'PTC'
ptc = ptc.groupby(['dataset', 'pretty_name'], as_index=False).mean()

df.drop(df.loc[df['dataset'].str.startswith("PTC")].index, inplace=True)
df = df.append(ptc, sort=False)

xticks = np.arange(len(df))

plt.bar(xticks, df['mean_acc'], yerr=df['acc_std'])
plt.plot(xticks, SOA.values(), 'r.')
plt.xticks(xticks, df['pretty_name'])
plt.savefig('exp3-result1.pdf')
plt.show()
