import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.activations import softmax
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from scipy.sparse import csr_matrix
from tensorflow import SparseTensor
import scipy

from model.GraphSAGELayer import GraphSAGELayer
from utils.graphs import get_data

# K.tf.enable_eager_execution()

s = scipy.sparse.random(8, 8, 0.5)

A, X, Y = get_data('MUTAG')
y = tf.convert_to_tensor(keras.utils.to_categorical(Y))

num_examples = len(A)
batch_size = len(A)
num_classes = 2

shape = (len(A), A[0].shape[-1], A[0].shape[-1])
print("shape:", shape)
indices = []
values = []

for i, a in enumerate(A):
    a = csr_matrix.tocoo(a)
    i_arr = np.repeat(np.array(i, dtype='int64'), a.row.shape)
    inds = np.stack([i_arr, a.row, a.col], axis=1)
    indices.append(inds)
    values.append(a.data)

print("eager:", tf.executing_eagerly())

indices = tf.convert_to_tensor(np.concatenate(indices), name="indices")
values = tf.convert_to_tensor(np.concatenate(values))
values = K.flatten(values)
shape = tf.convert_to_tensor(np.array(shape), dtype='int64')
shape = tf.reshape(shape, (3,))

print("indices.shape", indices.shape, indices.dtype)
print("values.shape", values.shape, values.dtype)
print("shape.shape", shape.shape)

a = tf.SparseTensor(indices, values, [188, 28, 28])
x = tf.convert_to_tensor(np.array(X))


# sparse_index = tf.placeholder(tf.int64, [None, 2])
#   sparse_ids = tf.placeholder(tf.int64, [None])
#   sparse_values = tf.placeholder(tf.float32, [None])
#   sparse_shape = tf.placeholder(tf.int64, [2])

A_in = Input(name='A_in', shape=(28, 28), sparse=True)
X_in = Input(name='X_in', shape=(28, 7))


def sparse_batch_generator(A_indices, A_values, A_shape, X, Y, batch_size):
    counter = 0
    total = num_examples
    print("BATCH GENERATOR")
    print("A_indices", A_indices)
    print("A_values", A_values)
    print("A_shape", A_shape)
    print("X", X)
    print("Y", Y)

    # print(A_shape)
    # a = SparseTensor(A_indices, A_values, shape)
    # print(a.dense_shape)

    while True:
        start = counter * batch_size
        remaining = total - start
        size = min(batch_size, remaining)
        counter = counter + 1

        if remaining < batch_size:
            counter = 0

        # a_slice = tf.sparse_slice(a, [], size)
        # a_indices = None

        yield [A_indices, A_values, A_shape, x], y

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

# if tf.executing_eagerly():
#     A_indices = Input(name='A_indices', tensor=indices)
#     A_values = Input(name='A_values', tensor=values)
#     A_shape = Input(name='A_shape', tensor=shape)
#     X_in = Input(name='X_in', tensor=x)
# else:
#     A_indices = Input(name='A_indices', shape=[None, 3], dtype='int64')
#     A_values = Input(name='A_values', shape=[None, 1])
#     A_shape = Input(name='A_shape', shape=[3], dtype='int64')
#     X_in = Input(name='X_in', shape=(28, 7))
#     A_in = Input(name='A_in', shape=[None, None, ])

z = GraphSAGELayer(aggregator=None,
                   neighbourhood_sampler=None,
                   n_neighbour_samples_per_node=2,
                   output_dim=1,
                   activation=softmax,
                   num_nodes=X[0].shape[0],
                   name='gs')([A_in, X_in])

z = keras.layers.Flatten(name='reshape')(z)
z = keras.layers.Dense(num_classes,
                       activation=keras.activations.softmax,
                       dtype='float32',
                       name='output')(z)

model = Model(inputs=[A_in, X_in], outputs=z)
model.compile(Adam(), loss=keras.losses.categorical_crossentropy)
model.summary()

# if tf.executing_eagerly():
#     model.fit(x=[indices, values, shape, x], y=y, steps_per_epoch=1, epochs=20)
# else:
#     model.fit(x=[indices, values, shape, x], y=y, epochs=1)
#     # model.fit(y=y, batch_size=batch_size, epochs=20)
#     model.fit_generator(generator=sparse_batch_generator(indices,
#                                                          values,
#                                                          shape,
#                                                          x, y,
#                                                          batch_size),
#                         steps_per_epoch=np.ceil(num_examples / batch_size),
#                         epochs=20)

model.train_on_batch(x=[a, x], y=y)

# loss, acc = model.evaluate(y=y, batch_size=batch_size)

# print('acc:', acc)



# model.fit_generator(steps_per_epoch=num_examples // batch_size)
