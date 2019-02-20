import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results-2019-01-30.csv', sep=";")
df = df.append(pd.read_csv('results-2019-01-31.csv', sep=";"))
df.sort_values(['classifier', 'exs_per_class'], inplace=True)

plt.title('Isomorphism Test')
cs = list(set(df['classifier'].values))
cs.remove('PiNet')
cs.sort()
cs.insert(0, 'PiNet')
for classifier in cs:

    data = df.loc[df['classifier'] == classifier]
    plt.plot(data['exs_per_class'], data['mean_acc'], '--x', label=classifier)

plt.ylabel('Mean Classification Accuracy')
plt.xlabel('No. of Training Examples per Class')
plt.legend()
plt.savefig('isomorphism-test.svg')
plt.savefig('isomorphism-test.pdf')
plt.show()
