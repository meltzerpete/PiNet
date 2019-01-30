# Graph Classifier

## Example Usage

```python
from model.GraphClassifier import GraphClassifier
from analysis.experiment2 import generate
from sklearn.model_selection import StratifiedKFold

num_classes = 3
batch_size = 5

A, X, Y = generate.get_tensors(num_nodes_per_graph=50,
                               num_graph_classes=num_classes,
                               num_node_classes=2,
                               num_graphs_per_class=100)

classifier = GraphClassifier()

folds = list(StratifiedKFold(n_splits=10, shuffle=True).split(X, Y))

# for evaluation
# A - List of adjacency matrices as ndarrays
# X - List of features matrices as ndarrays
# Y - (n x 1) ndarray containing class no.
# preprocess_A List of options as Strings
#   - 'add_self_loops'
#   - 'sym_normalise_A'
#   - 'laplacian'
#   - 'sym_norm_laplacian'
accs, times = classifier.fit_eval(A, X, Y, num_classes=num_classes,
                                  epochs=50, batch_size=batch_size, folds=folds, verbose=0)

# for predictions only
preds = classifier.get_predictions(A, X, Y, batch_size=batch_size)
```

## Computation

- `MyGCN` layer takes a list of `A_` and `X^{l}` as input, and gives a single output of `X^{l+1}` 

## Evaluation

StratifiedKFold:
- statified splits according to labels
- returns tuple of arrays of ids for train/test

## Data Generator

- [analysis/experiment2/generate.py](analysis/experiment2/generate.py)

# Experiments

## Experiment Dependencies

Add the following to the classpath:
- https://github.com/jajupmochi/py-graph
- [PyGamma](https://github.com/BraintreeLtd/PyGamma)

## Experiment 1: Message Passing Mechanisms

Observe effect of various matrices for message passing/diffusion.

- main: [model/ClassificationAccuracyTimeBenchmark.py](model/ClassificationAccuracyTimeBenchmark.py)
- data: [analysis/experiment1/results-2019-01-29.csv](analysis/experiment1/results-2019-01-29.csv)
- analysis: [analysis/experiment1/bar-charts.py](analysis/experiment1/bar-charts.py)

## Experiment 2: Isomorphism

- main (graph classifier): [analysis/eperiment2/IsomorphismExperiment.py](analysis/experiment2/IsomorphismExperiment.py)
- WL Kernel: [analysis/experiment2/WLKernel.py](analysis/experiment2/WLKernel.py)
- patchy-sans: TODO
- GCN only (with optional sum before dense): [analysis/experiment2/GCNWithOptionalSum.py](analysis/experiment2/GCNWithOptionalSum.py)
- data generator: [analysis/experiment2/generate.py](analysis/experiment2/generate.py)
- analysis: TODO

## Experiment 3: Benchmark Against SOA

- TODO