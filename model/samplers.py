import tensorflow as tf
from keras import backend as K


def gather_neighbour_features(row, X):
    def true_fn(X, neighbours_mask):
        # node has neighbours
        neighbours_idx = tf.where(neighbours_mask)
        return tf.gather(X, neighbours_idx)

    def false_fn(X):
        # node has no neighbours
        return tf.zeros((1, 1, tf.shape(X)[-1]), dtype='int64')

    neighbours_mask = tf.not_equal(row, tf.zeros_like(row))
    has_neighbours = tf.reduce_any(neighbours_mask)
    return tf.cond(has_neighbours,
                   true_fn=lambda: true_fn(X, neighbours_mask),
                   false_fn=lambda: false_fn(X))


def sample_neighbour_features(X_neighbours, num_samples):
    def true_fn(X, num_samples):
        # not enough neighbours -> pad
        num_rows = tf.shape(X)[0]
        diff = num_samples - num_rows
        pad = tf.pad(X, ((0, diff), (0, 0), (0, 0)))
        return pad

    def false_fn(X, num_samples, num_rows):
        # too many neighbours -> sample uniform random
        idx = K.arange(num_rows)
        shuffled_idx = tf.random_shuffle(idx)
        return tf.gather(X, shuffled_idx[:num_samples])

    num_rows = tf.shape(X_neighbours)[0]
    return tf.cond(num_rows <= num_samples,
                   true_fn=lambda: true_fn(X_neighbours, num_samples),
                   false_fn=lambda: false_fn(X_neighbours, num_samples, num_rows))


def aggregate_neighbour_features(X):
    # must return 1 row
    return K.max(X, axis=0)


def sample_and_aggregate_neighbours_features(row, X, num_samples):
    gathered = gather_neighbour_features(row, X)
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
        tf.enable_eager_execution()
        var = tf.contrib.eager.Variable
    else:
        var = tf.Variable

    A = var(A, dtype='int64', name='A_Adjacency_Matrix')
    X = var(X, dtype='int64', name='X_Features')
    N = var(2, dtype='int32', name='num_samples_per_node')

    out = tf.map_fn(lambda i: sample_and_aggregate_neighbours_features(A[i], X, N),
                    K.arange(0, tf.shape(A)[-1], 1, dtype='int64'))
    out = tf.reshape(out, (4, 2))

    if EAGER:
        print("out", out)
    else:
        k_eval = K.eval(out)
        print("out\n", k_eval)


EAGER = False
if __name__ == '__main__':
    main()
