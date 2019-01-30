import numpy as np
from sklearn.model_selection import StratifiedKFold
from model.GraphClassifier import GraphClassifier
from analysis.experiment2 import generate

num_nodes_per_graph=50
num_graph_classes=5
num_node_classes=2
num_graphs_per_class=50
batch_size = 5

A, X, Y = generate.get_tensors(num_nodes_per_graph,
                               num_graph_classes,
                               num_node_classes,
                               num_graphs_per_class)

classifier = GraphClassifier()

folds = list(StratifiedKFold(n_splits=10, shuffle=True).split(X, Y))
accs, times = classifier.fit_eval(A, X, Y, num_classes=num_graph_classes,
                                  epochs=200, batch_size=batch_size, folds=folds, verbose=0)

# preds = classifier.get_predictions(A, X, Y, batch_size=batch_size)

print()
print("accs:", accs)
print("tmes:", times)
print("mean acc:", np.mean(accs))
print("mean time:", np.mean(times))
print()
