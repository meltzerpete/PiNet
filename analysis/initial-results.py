import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('initial-results.csv', sep='|', header=1, names=['Dataset', 'Ratio', 'Acc', 'Std', 'SOA'])
plt.bar(np.arange(3), df['Acc'], yerr=df['Std'])
plt.xticks(np.arange(3), df['Dataset'])
plt.plot(np.arange(3), df['SOA'], "r.")
plt.title("Mean Classification Accuracy")
plt.savefig('graph-classifier-initial-results.pdf')
plt.show()
