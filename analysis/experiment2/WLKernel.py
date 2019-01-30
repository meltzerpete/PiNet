import numpy as np
import networkx as nx
from sklearn.model_selection import StratifiedShuffleSplit

from analysis.experiment2 import generate
from pygraph.kernels.weisfeilerLehmanKernel import weisfeilerlehmankernel
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# using code from https://github.com/jajupmochi/py-graph
class WLKernel:

    def name(self):
        return 'WLKernel'

    def get_accuracies(self, A, X, Y, num_graph_classes, splits=None, batch_size=None):
        G = [nx.from_numpy_array(a) for a in A]
        for g, x in zip(G, X):
            _x = np.argmax(x)
            nx.set_node_attributes(g, _x, 'node_label')

        wl = weisfeilerlehmankernel(G, node_label='node_label')
        plt.imshow(wl[0])
        plt.show()

        km = wl[0]

        svc = SVC(kernel='precomputed')

        accs = []
        for train_ids, test_ids in splits:
            K_train = km[np.ix_(train_ids, train_ids)]
            K_test = km[np.ix_(test_ids, train_ids)]

            svc.fit(K_train, Y.flatten()[train_ids])

            preds = svc.predict(K_test)
            acc = accuracy_score(Y[test_ids], preds)
            accs.append(acc)
            # print("acc:", acc)

        # print("mean acc:", np.mean(accs))
        # print("std:", np.std(accs))
        return accs


if __name__ == '__main__':
    num_graph_classes = 5
    A, X, Y = generate.get_tensors(50, num_graph_classes, 2, 100)
    splits = StratifiedShuffleSplit(10, test_size=num_graph_classes * 10).split(X, Y)
    WLKernel().get_accuracies(A, X, Y, num_graph_classes, splits)
