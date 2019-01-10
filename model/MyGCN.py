from keras import backend as K
from keras import activations, initializers, regularizers
from keras.engine.topology import Layer


class MyGCN(Layer):

    def __init__(self, output_dim, activation=None, **kwargs):
        self.units = output_dim
        super(MyGCN, self).__init__(**kwargs)
        self.activation = activations.get(activation)

    def build(self, input_shape):
        # print("input: ", input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[2] - input_shape[1], self.units),
                                      initializer=initializers.orthogonal(),
                                      regularizer=regularizers.l2(5e-4))

        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer=initializers.zeros())
        super(MyGCN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        A_ = inputs[:, :, :inputs.shape[1]]
        # print("A_: ", A_.shape)
        X = inputs[:, :, inputs.shape[1]:]
        # print("X: ", X.shape)
        W = self.kernel
        # print("W: ", W.shape)
        XW = K.dot(X, W)
        # print("XW: ", XW.shape)
        A_XW = K.batch_dot(A_, XW)
        # print("A_XW: ", A_XW.shape)
        # print("B: ", self.bias.shape)
        out = A_XW + self.bias
        # print("out: ", out.shape)
        Y = K.concatenate([A_, out], axis=2)
        # print("Y:", Y.shape)
        return self.activation(Y)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[1] + self.units

