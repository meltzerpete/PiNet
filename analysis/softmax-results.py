import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

SOA = {
    'MUTAG': 88.9,
    'PTC': 69.0,
    'PROTEINS': 76.4,
    'NCI1': 82.7,
    'NCI109': 82.4,
    'DD': 79.8,
    # 'ENZYMES': 61.8,
}

soa = pd.DataFrame.from_dict(SOA, orient='index').reset_index()
soa.columns = ['dataset', 'soa_acc']
soa = soa.sort_values('dataset').reset_index(drop=True)
soa['color'] = soa.index.map(lambda x: 'C' + str(x))

df = pd.read_csv('withA_.csv', sep=';')
ptc = df.loc[df.dataset.str.contains('PTC')].groupby('batch_size').mean().reset_index()
ptc['dataset'] = 'PTC'

df = df.append(ptc, sort=False)
df = df.loc[df.dataset.isin(SOA.keys())]
df = df.merge(soa, 'outer', 'dataset')
df = df.sort_values(['dataset', 'batch_size'])

labels = df.dataset + ' (' + df.batch_size.map(str) + ')'

plt.subplot()
x_pos = np.arange(len(df))
plt.bar(x_pos, df['mean_acc'], yerr=df['acc_std'], color=df['color'])
plt.xticks(x_pos, labels, rotation=90)
plt.xlabel('Dataset (batch size)')
plt.plot(x_pos, df['soa_acc'] / 100, "r.")
plt.title("Mean Classification Accuracy")
plt.tight_layout()

plt.savefig('withA_-accuracy.pdf')
plt.show()

plt.subplot()
x_pos = np.arange(len(df))
plt.bar(x_pos, df['mean_train_time(s)'], yerr=df['time_std'], color=df['color'])
plt.xticks(x_pos, labels, rotation=90)
plt.xlabel('Dataset (batch size)')
plt.yscale('log')
plt.title("Mean Training Time (s)")
plt.tight_layout()

plt.savefig('withA_-time.pdf')
plt.show()

df = df.drop(columns=['all_accs', 'all_times', 'color'])
df.columns = ['Dataset', 'Batch Size', 'Mean Acc.', 'Std.', 'Mean Training Time (s)', 'Std.', 'SOA']
print(df.to_csv(index=False))
