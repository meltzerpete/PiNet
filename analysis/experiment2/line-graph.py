import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from scipy.stats import ttest_ind

df = pd.read_csv('results-2019-01-30.csv', sep=";")
df = df.append(pd.read_csv('results-2019-01-31.csv', sep=";"))
df.sort_values(['classifier', 'exs_per_class'], inplace=True)

df['accs'] = df['accs'].apply(literal_eval)
pinet = df.loc[df['classifier'] == 'PiNet']

plt.title('Isomorphism Test')
cs = list(set(df['classifier'].values))
cs.remove('PiNet')
cs.sort()
cs.insert(0, 'PiNet')
for classifier in cs:

    data = df.loc[df['classifier'] == classifier]
    plt.plot(data['exs_per_class'], data['mean_acc'], '--x', label=classifier)

    if classifier != 'PiNet':
        for i in range(10):
            p_value = ttest_ind(pinet['accs'][i],
                                data['accs'].loc[data['exs_per_class'] == 2 * i + 2].any(),
                                equal_var=False)[1]
            print(classifier, i, p_value, p_value < 0.05)

marker_style = dict(color='red', linestyle=':', marker='o',
                    markersize=15, markerfacecoloralt='gray')
plt.plot(6, 0.504, **marker_style, fillstyle='none')
plt.ylabel('Mean Classification Accuracy')
plt.xlabel('No. of Training Examples per Class')
plt.legend(bbox_to_anchor=(0.9, 0.65))
plt.savefig('isomorphism-test.svg')
plt.savefig('isomorphism-test.pdf')
plt.show()
