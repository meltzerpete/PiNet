from keras import backend as K


def gather_neighbour_features(A, X, i):
    def true_fn(X, neighbours_mask):
        # node has neighbours
        neighbours_idx = K.tf.where(neighbours_mask)
        # print("where", neighbours_idx)
        gather = K.gather(X, neighbours_idx)
        # print("gather", gather)
        return gather

    def false_fn(X):
        # node has no neighbours
        return K.zeros((1, X.shape[-1]), dtype='int64')

    row = A[i]
    neighbours_mask = K.not_equal(row, K.zeros_like(row))
    has_neighbours = K.any(neighbours_mask)
    return K.tf.cond(has_neighbours,
                     true_fn=lambda: true_fn(X, neighbours_mask),
                     false_fn=lambda: false_fn(X))


def sample_neighbour_features(X, num_samples):
    def true_fn(X):
        zeros = K.zeros((num_samples, 1, X.shape[-1]), dtype='int64')
        return K.tf.add(zeros, X)

    def false_fn(X):
        idx = K.arange(num_rows)
        shuffled_idx = K.tf.random_shuffle(idx)
        return K.gather(X, shuffled_idx[:num_samples])

    num_rows = K.tf.shape(X)[0]
    return K.tf.cond(num_rows <= num_samples,
                     true_fn=lambda: true_fn(X),
                     false_fn=lambda: false_fn(X))


def aggregate_neighbour_features(X):
    # must return 1 row
    print("X\n", X)
    print("MAX:\n", K.max(X, axis=0))
    return K.max(X, axis=0)


def sample_and_aggregate_neighbours_features(A, X, i):
    gathered = gather_neighbour_features(A, X, i)
    sampled = sample_neighbour_features(gathered, 2)
    aggregated = aggregate_neighbour_features(sampled)
    return aggregated


def main():
    A = [[1, 1, 1, 1],
         [1, 1, 1, 0],
         [1, 1, 0, 0],
         [0, 0, 0, 0]]

    X = [[10, 11],
         [20, 21],
         [30, 31],
         [40, 41]]

    K.tf.enable_eager_execution()

    A = K.tf.Variable(A, dtype='int64')
    X = K.tf.Variable(X, dtype='int64')

    # print("A.shape", A.shape)
    # print("X.shape", X.shape)

    # AGG
    n = K.arange(0, 4, 1, dtype='int64')
    out = K.tf.map_fn(lambda i: sample_and_aggregate_neighbours_features(A, X, i),
                      n, infer_shape=False)
    print("out", out)


if __name__ == '__main__':
    main()
