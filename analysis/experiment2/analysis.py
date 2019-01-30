import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results-2019-01-30.csv', sep=";")

plt.title('Isomorphism Test')
for classifier in set(df['classifier'].values):

    data = df.loc[df['classifier'] == classifier]
    plt.plot(data['exs_per_class'], data['mean_acc'], label=classifier)

plt.ylabel('Mean Classification Accuracy')
plt.xlabel('No. of Training Examples per Class')
plt.legend()
plt.savefig('isomorphism-test.pdf')
plt.show()
