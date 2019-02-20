import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv', delimiter=';')
ptc = df.loc[df['pretty_name'].str.startswith("PTC")]
ptc = ptc.assign(pretty_name='PTC')
ptc = ptc.groupby(['pretty_name', 'classifier'], as_index=False).mean()

df.drop(df.loc[df['pretty_name'].str.startswith("PTC")].index, inplace=True)
df = df.append(ptc, sort=False)

df.sort_values('pretty_name', inplace=True)

df['print_acc'] = df[['mean_acc', 'acc_std']].apply(lambda row: f'${row[0]:.2f} \pm {row[1]:.2f}$', axis=1)

acc_pivot_table = df.pivot('classifier', columns='pretty_name', values='print_acc')

print(acc_pivot_table.to_latex(escape=False))
