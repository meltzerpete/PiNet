import numpy as np
from sklearn.model_selection import StratifiedKFold
from model.GraphClassifier import GraphClassifier
from analysis.experiment2 import generate

num_classes = 3
batch_size = 5

A, X, Y = generate.get_tensors(num_nodes_per_graph=5,
                               num_graph_classes=num_classes,
                               num_node_classes=2,
                               num_graphs_per_class=10)

classifier = GraphClassifier(A, X, Y, 'generated', num_classes=num_classes)

classifier.preprocess_A()

folds = list(StratifiedKFold(n_splits=10, shuffle=True).split(X, Y))
accs, times = classifier.build_fit_eval(epochs=50, batch_size=batch_size, folds=folds, verbose=0)

preds = classifier.get_predictions(A, X, Y, batch_size=batch_size)

print()
print("accs:", accs)
print("tmes:", times)
print("mean acc:", np.mean(accs))
print("mean time:", np.mean(times))
print()
