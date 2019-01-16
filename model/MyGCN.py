from keras import backend as K
from keras import activations, initializers, regularizers
from keras.engine.topology import Layer
import tensorflow as tf


class MyGCN(Layer):

    def __init__(self, output_dim, activation=None, **kwargs):
        self.units = output_dim
        super(MyGCN, self).__init__(**kwargs)
        self.activation = activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1][2], self.units),
                                      initializer=initializers.orthogonal(),
                                      regularizer=regularizers.l2(5e-4))

        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer=initializers.zeros())
        super(MyGCN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        A_ = inputs[0]
        X = inputs[1]
        W = self.kernel
        XW = K.dot(X, W)
        A_XW = K.batch_dot(A_, XW)
        out = A_XW + self.bias
        return self.activation(out)

    def compute_output_shape(self, input_shape):
        return input_shape[1][0], input_shape[1][1], self.units
