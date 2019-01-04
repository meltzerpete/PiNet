# Graph Classifier


- borrowing code from PyGamma for data import etc.
- using dense tensor for now - worry about this later


## Preprocessing

- calculate `A_` tensor using GCN code from PyGamma (requires sparse matrix - convert to sparse and back for now)
- `G` <- concat `A_` (N x n x n) and `X` (N x n x d) on `axis=2` to get (N x n x (n+d)) tensor


## Evaluation

StratifiedKFold:
- statified splits according to labels
- returns tuple of arrays of ids for train/test

## Next Steps

- GCN-GC layer
