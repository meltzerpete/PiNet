from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import Orthogonal


class MyGCN(Layer):

    def __init__(self, output_dim, **kwargs):
        self.units = output_dim
        super(MyGCN, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[2] - input_shape[1], self.units),
                                      initializer=Orthogonal(),
                                      trainable=True)
        super(MyGCN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        A_ = inputs[:, :, :inputs.shape[1]]
        print("A_: ", A_.shape)
        X = inputs[:, :, inputs.shape[1]:]
        print("X: ", X.shape)
        W = self.kernel
        print("W: ", W.shape)
        XW = K.dot(X, W)
        print("XW: ", XW.shape)
        A_XW = K.batch_dot(A_, XW)
        print("A_XW: ", A_XW.shape)
        Y = K.concatenate([A_, A_XW], axis=2)
        print("Y:", Y.shape)
        return Y

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[1] + self.units

