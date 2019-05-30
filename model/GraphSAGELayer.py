from keras import backend as K
from keras import activations, initializers, regularizers
from keras.layers import Layer
from tensorboard import summary
from tensorflow import SparseTensor


class GraphSAGELayer(Layer):

    def __init__(self,
                 aggregator,
                 neighbourhood_sampler,
                 n_neighbour_samples_per_node,
                 output_dim,
                 num_nodes,
                 activation=activations.relu,
                 **kwargs):
        super(GraphSAGELayer, self).__init__(**kwargs)
        self.aggregator = aggregator
        self.neighbourhood_sampler = neighbourhood_sampler
        self.n_neighbour_samples_per_node = n_neighbour_samples_per_node
        self.output_dim = output_dim
        self.activation = activation
        self.num_nodes = num_nodes
        self.agg_weights = None
        self.agg_bias = None

    def build(self, input_shape):
        n, r, c = input_shape[-1]

        self.agg_weights = self.add_weight(name='aggWeights',
                                           shape=[c * 2, self.output_dim],
                                           initializer=initializers.glorot_uniform())
        self.agg_bias = self.add_weight('aggBias',
                                        shape=[self.output_dim],
                                        initializer=initializers.zeros())

    def call(self, inputs, **kwargs):
        A, X = inputs

        def slice_sparse_matrix(all_A, i):
            n = K.tf.shape(all_A)[-1]
            ret = K.tf.sparse.slice(all_A, (i, 0, 0), (1, n, n))
            reshaped = K.tf.sparse_reshape(ret, [n, n])
            return reshaped

        def slice_matrix(all_X, i):
            n = K.tf.shape(X)[1]
            m = K.tf.shape(X)[2]
            slice = K.tf.slice(all_X, (i, 0, 0), (1, n, m))
            reshaped = K.tf.reshape(slice, [n, m])
            return reshaped

        h = K.tf.map_fn(
            lambda i: self.per_A_X(slice_sparse_matrix(A, i), slice_matrix(X, i)),
            K.arange(0, K.tf.shape(X)[0], 1), infer_shape=False, dtype=K.dtype(X))

        # normalise


        h.set_shape([X.shape[0], self.num_nodes, self.output_dim])

        return K.tf.cast(h, dtype='float32')

    def per_A_X(self, A, X):
        def slice_row(A, i):
            n = K.tf.shape(A)[-1]
            ret = K.tf.sparse.slice(A, (i, 0), (1, n))
            return ret

        out = K.tf.map_fn(lambda i: self._per_V(i, slice_row(A, i), X,
                                                self.n_neighbour_samples_per_node),
                          K.arange(0, K.tf.shape(X)[-2], 1), infer_shape=False, dtype=K.dtype(X))


        weights = K.tf.cast(self.agg_weights, K.dtype(X))
        bias = K.tf.cast(self.agg_bias, K.dtype(X))
        out = K.tf.reshape(out, [K.tf.shape(X)[-2], K.tf.shape(X)[-1] * 2])
        out = K.tf.matmul(out, weights)
        out = K.bias_add(out, bias)

        out = activations.relu(out)

        # out.csv = K.tf.reshape(out.csv, (K.tf.shape(X)[-2], self.output_dim))
        return out

    def compute_output_shape(self, input_shape):
        n, r, c = input_shape[-1]
        return n, r, self.output_dim

    def _per_V(self, i, row, X, num_samples):
        row = K.tf.sparse.to_dense(row)
        reshaped = K.tf.reshape(row, [K.tf.shape(X)[-2]])
        gathered = GraphSAGELayer._gather_neighbour_features(reshaped, X)
        sampled = GraphSAGELayer._sample_neighbour_features(gathered, num_samples)
        aggregated = self._aggregate_neighbour_features(sampled)
        v_prev = K.slice(X, [i, 0], [1, K.tf.shape(X)[-1]])
        out = K.concatenate([v_prev, aggregated])
        return out

    @staticmethod
    def _gather_neighbour_features(adj_row, X_full):
        def true_fn(X_full, neighbours_mask):
            # node has neighbours
            neighbours_idx = K.tf.where(neighbours_mask)
            ret = K.tf.gather(X_full, neighbours_idx)
            return ret

        def false_fn(X_full):
            # node has no neighbours
            ret = K.tf.zeros((1, 1, K.tf.shape(X_full)[-1]), dtype=K.dtype(X_full))
            return ret

        neighbours_mask = K.tf.not_equal(adj_row, K.tf.zeros_like(adj_row))
        has_neighbours = K.tf.reduce_any(neighbours_mask)
        return K.tf.cond(has_neighbours,
                         true_fn=lambda: true_fn(X_full, neighbours_mask),
                         false_fn=lambda: false_fn(X_full))

    @staticmethod
    def _sample_neighbour_features(X_neighbours, num_samples):
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
            return K.tf.gather(X, shuffled_idx[:num_samples])

        num_rows = K.tf.shape(X_neighbours)[0]
        return K.tf.cond(num_rows <= num_samples,
                         true_fn=lambda: true_fn(X_neighbours, num_samples),
                         false_fn=lambda: false_fn(X_neighbours, num_samples, num_rows))

    def _aggregate_neighbour_features(self, X):
        # must return 1 row
        return K.max(X, axis=0)
