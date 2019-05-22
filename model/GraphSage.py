import numpy as np

from model.GraphSAGELayer import GraphSAGELayer
from utils.graphs import get_data
from scipy.sparse import csr_matrix

import keras
import keras.backend as K
import tensorflow as tf
from keras.activations import softmax
from keras.layers import Input, Dot, Reshape, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam

A, X, Y = get_data('MUTAG')
y = keras.utils.to_categorical(Y)

num_examples = len(A)
batch_size = len(A)
num_classes = y.shape[-1]

shape = (len(A), A[0].shape[-1], A[0].shape[-1])
indices = []
values = []

for i, a in enumerate(A):
    a = csr_matrix.tocoo(a)
    i_arr = np.repeat(i, a.row.shape)
    inds = np.stack([i_arr, a.row, a.col], axis=1)
    indices.append(inds)
    values.append(a.data)

indices = np.concatenate(indices)
values = np.concatenate(values)

# K.tf.enable_eager_execution()

a = K.tf.SparseTensor(indices, values, shape)
x = tf.convert_to_tensor(np.array(X))

A_in = Input(name='A_in', tensor=tf.cast(a, 'float32'))
X_in = Input(name='X_in', tensor=tf.cast(x, 'float32'))

# A_in = Input(name='A_in', sparse=True, shape=(28, 28))
# X_in = Input(name='X_in', shape=(28, 7))

h = GraphSAGELayer(aggregator=None,
                   neighbourhood_sampler=None,
                   n_neighbour_samples_per_node=2,
                   output_dim=1,
                   activation=softmax,
                   num_nodes=X[0].shape[0],
                   name='gs')([A_in, X_in])

reshape = keras.layers.Reshape([X[0].shape[-2] * X[0].shape[-1]])(h)
out = keras.layers.Dense(num_classes, activation=keras.activations.softmax, dtype='float64')(reshape)

model = Model(inputs=[A_in, X_in], outputs=out)
model.compile(Adam(), loss=keras.losses.categorical_crossentropy, metrics=['categorical_accuracy'])
model.summary()

model.fit(y=y, batch_size=batch_size, epochs=20)

loss, acc = model.evaluate(y=y, batch_size=batch_size)

print('acc:', acc)


def batch_generator(A, X, Y, batch_size):
    counter = 0
    total = 188

    indices = []
    values = []

    for i, a in enumerate(A):
        a = csr_matrix.tocoo(a)
        i_arr = np.repeat(i, a.row.shape)
        inds = np.stack([i_arr, a.row, a.col], axis=1)
        indices.append(inds)
        values.append(a.data)

    while True:
        start = counter * batch_size
        remaining = total - start
        size = min(batch_size, remaining)
        counter = counter + 1

        if remaining < batch_size:
            counter = 0

        shape = (size, A[0].shape[-1], A[0].shape[-1])
        a = K.tf.SparseTensor(indices=np.concatenate(indices[start:start + size]),
                              values=np.concatenate(values[start:start + size]),
                              dense_shape=shape)

        x = tf.convert_to_tensor(np.array(X[start:start + size]))
        y = keras.utils.to_categorical(Y[start:start + size])

        a = tf.cast(a, dtype='float32')
        x = tf.cast(x, dtype='float32')

        yield [a, x], y



# model.fit_generator(steps_per_epoch=num_examples // batch_size)
