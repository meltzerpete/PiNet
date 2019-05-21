import tensorflow as tf
from keras import backend as K


def gather_neighbour_features(A, X, i):
    def true_fn(X, neighbours_mask):
        # node has neighbours
        neighbours_idx = K.tf.where(neighbours_mask)
        return K.gather(X, neighbours_idx)

    def false_fn(X):
        # node has no neighbours
        return K.tf.zeros((1, 1, K.tf.shape(X)[-1]), dtype='int64')

    row = A[i]
    neighbours_mask = K.not_equal(row, K.zeros_like(row))
    has_neighbours = K.any(neighbours_mask)
    return K.tf.cond(has_neighbours,
                     true_fn=lambda: true_fn(X, neighbours_mask),
                     false_fn=lambda: false_fn(X))


def sample_neighbour_features(X, num_samples):
    def true_fn(X, num_samples):
        # not enough neighbours -> pad
        num_rows = K.tf.shape(X)[0]
        diff = num_samples - num_rows
        pad = K.tf.pad(X, ((0, diff), (0, 0), (0, 0)))
        return pad

    def false_fn(X, num_samples, num_rows):
        # too many neighbours -> sample uniform random
        idx = K.arange(num_rows)
        shuffled_idx = K.tf.random_shuffle(idx)
        return K.gather(X, shuffled_idx[:num_samples])

    num_rows = K.tf.shape(X)[0]
    return K.tf.cond(num_rows <= num_samples,
                     true_fn=lambda: true_fn(X, num_samples),
                     false_fn=lambda: false_fn(X, num_samples, num_rows))


def aggregate_neighbour_features(X):
    # must return 1 row
    return K.max(X, axis=0)


def sample_and_aggregate_neighbours_features(A, X, num_samples, i):
    gathered = gather_neighbour_features(A, X, i)
    sampled = sample_neighbour_features(gathered, num_samples)
    aggregated = aggregate_neighbour_features(sampled)
    return aggregated


def main():
    sess = tf.Session()
    K.set_session(sess)

    A = [[1, 1, 1, 1],
         [1, 1, 1, 0],
         [1, 1, 0, 0],
         [0, 0, 0, 1]]

    X = [[10, 11],
         [20, 21],
         [30, 31],
         [40, 41]]

    if EAGER:
        K.tf.enable_eager_execution()
        var = K.tf.contrib.eager.Variable
    else:
        var = K.tf.Variable

    A = var(A, dtype='int64', name='A_Adjacency_Matrix')
    X = var(X, dtype='int64', name='X_Features')
    N = var(2, dtype='int32', name='num_samples_per_node')

    out = K.tf.map_fn(lambda i: sample_and_aggregate_neighbours_features(A, X, N, i),
                      K.arange(0, K.tf.shape(A)[-1], 1, dtype='int64'))
    out = K.reshape(out, (4, 2))

    if EAGER:
        print("out", out)
    else:
        print("X\n", K.eval(X))
        k_eval = K.eval(out)
        print("out\n", k_eval)
        print(k_eval.shape)


EAGER = False
if __name__ == '__main__':
    main()
