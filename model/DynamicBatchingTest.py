from ast import literal_eval

import keras
from keras import backend as k
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam
from scipy.sparse import csr_matrix
import scipy.sparse as sparse


class SparseLayer(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(SparseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SparseLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        x2 = tf.matmul(x, k.transpose(x))
        return k.slice(x2, [0, 2], [4, 2])

    def compute_output_shape(self, input_shape):
        # N x 1
        return input_shape[0], 2


x1 = [
    [1., 0.],
    [1., 0.]
]
x2 = [
    [0., 1.],
    [0., 1.]
]

xs = np.concatenate([x1, x2], axis=0)
mmul = np.matmul(xs, xs.transpose())
print(mmul)

x = list(map(csr_matrix, [x1, x2]))
x = sparse.vstack(x, format='csr')

y = np.array([0, 0, 1, 1])
y = 1 - y
import pandas as pd

# df = pd.DataFrame(x)

# d = keras.preprocessing.image.ImageDataGenerator()

# x = tf.Tensor(x, shape=[4, None])
#
# tf.placeholder('int32', shape=[None, None])
# tf.convert_to_tensor(x)

y = keras.utils.to_categorical(y)

X_in = keras.layers.Input(shape=[None])
h = SparseLayer()(X_in)
out = keras.layers.Lambda(lambda x: x, name='final')(h)

model = keras.models.Model(inputs=X_in, outputs=out)

model.summary()
model.compile(Adam(), loss=keras.losses.mse, metrics=['accuracy'])

preds = model.predict(x)
print(preds)

# model.fit(x, y, epochs=5)
