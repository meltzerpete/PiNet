from keras import backend as K
from keras import activations, initializers, regularizers
from keras.engine.topology import Layer
from tensorboard import summary


class MyGCN(Layer):

    def __init__(self, output_dim, activation=activations.relu, learn_pqr=False, **kwargs):
        self.units = output_dim
        super(MyGCN, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.learn_pqr = learn_pqr

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1][2], self.units),
                                      initializer=initializers.orthogonal(),
                                      regularizer=regularizers.l2(5e-4))

        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer=initializers.zeros())
        if self.learn_pqr:
            self.p = self.add_weight(name='p',
                                     shape=(1,),
                                     initializer=initializers.constant(0))
            self.q = self.add_weight(name='q',
                                     shape=(1,),
                                     initializer=initializers.constant(0))

        super(MyGCN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        A = inputs[0]
        X = inputs[1]

        # print("A:", A.shape)
        dims = int(A.shape[1])

        if self.learn_pqr:
            p = activations.sigmoid(self.p)
            q = activations.sigmoid(self.q)

            I = K.eye(dims)

            Dr = K.sum(A, axis=1)
            k_vec = p * K.ones_like(Dr) + (1 - p) * Dr
            k_inv_root = K.pow(K.sqrt(k_vec), -.5)
            mask = K.tf.is_inf(k_inv_root)
            k_clean = K.tf.where(mask, K.tf.zeros_like(k_inv_root), k_inv_root)

            D = K.tf.matrix_diag(k_clean)

            A_ = K.batch_dot(K.batch_dot(D, (A + q * I)), D)
        else:
            A_ = A

        W = self.kernel
        XW = K.dot(X, W)
        A_XW = K.batch_dot(A_, XW)
        out = A_XW + self.bias
        K.tf.verify_tensor_all_finite(out, "out contains infs")
        return self.activation(out)

    def compute_output_shape(self, input_shape):
        return input_shape[1][0], input_shape[1][1], self.units
