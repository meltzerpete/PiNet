import numpy as np
import networkx as nx
from sklearn.model_selection import StratifiedKFold

from pygraph.kernels.weisfeilerLehmanKernel import weisfeilerlehmankernel
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import math

# using code from https://github.com/jajupmochi/py-graph

num_classes = 2
batch_size = 5

A, X, Y = pickle.load(open('isoGraphs2classes50nodes5ncs100graphs.p', 'rb'))

G = [nx.from_numpy_array(a) for a in A]
for g, x in zip(G, X):
    _x = np.argmax(x)
    nx.set_node_attributes(g, _x, 'node_label')

wl = weisfeilerlehmankernel(G, node_label='node_label')
plt.imshow(wl[0])
plt.show()

km = wl[0]

svc = SVC(kernel='precomputed')

folds = list(StratifiedKFold(n_splits=10, shuffle=True).split(X, Y))

accs = []
for train_ids, test_ids in folds:
    K_train = km[np.ix_(train_ids, train_ids)]
    K_test = km[np.ix_(test_ids, train_ids)]

    svc.fit(K_train, Y.flatten()[train_ids])

    preds = svc.predict(K_test)
    acc = accuracy_score(Y[test_ids], preds)
    accs.append(acc)
    print("acc:", acc)

print("mean acc:", np.mean(accs))
print("std:", np.std(accs))
