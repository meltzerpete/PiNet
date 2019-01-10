import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('initial-results-learn-matrix.csv', sep='|', header=1,
                 names=['Dataset', 'Ratio', 'Acc', 'Std', 'Acc-matrix', 'Std-matrix', 'SOA'])
plt.bar(np.arange(3), df['Acc'], yerr=df['Std'])
plt.xticks(np.arange(6), df['Dataset'].append(df['Dataset']))
plt.plot(np.arange(3), df['SOA'], "r.")
plt.bar(np.arange(3, 6), df['Acc-matrix'], yerr=df['Std-matrix'])
# plt.xticks(np.arange(3, 6), df['Dataset'])
plt.plot(np.arange(3, 6), df['SOA'], "r.")
plt.title("Mean Classification Accuracy")
plt.savefig('graph-classifier-learn-matrix.pdf')
plt.show()
