# Graph Classifier


- borrowing code from PyGamma for data import etc.


## Example Usage

```python
# A - list of adjacency matrices as ndarrays
# X - list of features matrices as ndarrays
# Y - (n x 1) ndarray containing class no.

classifier = GraphClassifier(A, X, Y, 'dataset_name', num_classes=num_classes)

# preprocess_A() converts A to list of scipy.sparse.csr_matrix - this must be called
# optionally pass list of preprocessing methods from:
# - add self loops
# - sym_normalise_A
# - laplacian
# - sym_norm_laplacian

classifier.preprocess_A(['add_self_loops', 'sym_normalise_A'])

folds = list(StratifiedKFold(n_splits=10, shuffle=True).split(X, Y))

# for evaluation
accs, times = classifier.build_fit_eval(epochs=50, batch_size=batch_size, folds=folds, verbose=0)

# for predictions only
preds = classifier.get_predictions(A, X, Y, batch_size=batch_size)
```

## Computation

- `MyGCN` layer takes a list of `A_` and `X^{l}` as input, and gives a single output of `X^{l+1}` 

## Evaluation

StratifiedKFold:
- statified splits according to labels
- returns tuple of arrays of ids for train/test
