from keras import backend as K
from keras import activations, initializers, regularizers
from keras.layers import Layer
from tensorboard import summary


class GraphSAGELayer(Layer):

    def __init__(self, aggregator, neighbourhood_sampler, output_dim, activation=activations.relu, **kwargs):
        super(GraphSAGELayer, self).__init__(**kwargs)
        self.aggregator = aggregator
        self.neighbourhood_sampler = neighbourhood_sampler
        self.output_dim = output_dim
        self.activation = activation

    def build(self, input_shape):
        # self.kernel = self.add_weight(name='kernel')
        pass

    def call(self, inputs, **kwargs):
        A, X = inputs

        # aggregate features
        # h_N = self.aggregator(self.neighbourhood_sampler)
        h_N = X

        # concat
        h_v = K.concatenate((X, h_N), axis=-1)
        print(h_v.shape)

        # apply dense layer
        # regularise
        return X

    def compute_output_shape(self, input_shape):
        return input_shape
