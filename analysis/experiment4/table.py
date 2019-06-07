import pandas as pd
from scipy.stats import ttest_ind_from_stats

df = pd.read_csv('results.csv', delimiter=';')
ptc = df.loc[df['pretty_name'].str.startswith("PTC")]
ptc = ptc.assign(pretty_name='PTC')
ptc = ptc.groupby(['pretty_name', 'classifier'], as_index=False).mean()

df.drop(df.loc[df['pretty_name'].str.startswith("PTC")].index, inplace=True)
df = df.append(ptc, sort=False)

df.sort_values('pretty_name', inplace=True)

## T TEST
# df['ttest_PiNet_l'] = df[['mean_acc', 'acc_std']]
pinet_l_mean = df.loc[df['classifier'] == 'GC'][['pretty_name', 'mean_acc', 'acc_std']]
with_pinet = df.merge(pinet_l_mean, how='outer', on='pretty_name').drop(columns=['dataset', 'all_accs', 'all_times'])
with_pinet['p_with_l'] = with_pinet.apply(
    lambda row: ttest_ind_from_stats(
        row['mean_acc_y'], row['acc_std_y'], 10,
        row['mean_acc_x'], row['acc_std_x'], 10,
        equal_var=False)[1], axis=1)

pinet_m_mean = df.loc[df['classifier'] == 'GCp1q0'][['pretty_name', 'mean_acc', 'acc_std']]
with_pinet_m = df.merge(pinet_m_mean, how='outer', on='pretty_name').drop(columns=['dataset', 'all_accs', 'all_times'])
with_pinet_m['p_with_m'] = with_pinet_m.apply(
    lambda row: ttest_ind_from_stats(
        row['mean_acc_y'], row['acc_std_y'], 10,
        row['mean_acc_x'], row['acc_std_x'], 10,
        equal_var=False)[1], axis=1)

# trues = df.loc[(df['p_with_l'] < 0.05) | (df['p_with_m'] < 0.05)]
sorted_l = with_pinet.sort_values('p_with_l')
sorted_m = with_pinet_m.sort_values('p_with_m')

df['print_acc'] = df[['mean_acc', 'acc_std']].apply(lambda row: f'${row[0]:.2f} \pm {row[1]:.2f}$', axis=1)
acc_pivot_table = df.pivot('classifier', columns='pretty_name', values='print_acc')
print(acc_pivot_table.to_latex(escape=False))
