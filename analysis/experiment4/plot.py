from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('out.csv', sep=';', skiprows=10)
df = df.loc[df['classifier'] != 'classifier']
df['mean_acc'] = df['mean_acc'].astype('float')
df['acc_std'] = df['acc_std'].astype('float')
df['all_accs'] = df['all_accs'].apply(literal_eval)

df.reset_index()

plt.title('Mean Classification Accuracy\n10-Fold Cross Validation')

# plt.bar(np.arange(len(df)), df['mean_acc'], yerr=df['acc_std'])
plt.boxplot(df['all_accs'])
# plt.errorbar(np.arange(len(df)), df['mean_acc'], df['acc_std'])

names = df[['classifier', 'pretty_name']].apply(lambda x: x[0] + '\n' + x[1], axis=1)

plt.xticks(np.arange(1, len(df) + 1), names, rotation='45')
plt.tight_layout()

plt.savefig('plot.pdf')
plt.savefig('plot.png')
plt.show()
