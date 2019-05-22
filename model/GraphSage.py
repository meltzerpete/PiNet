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

shape = (len(A), A[0].shape[-1], A[0].shape[-1])
indices = []
values = []

for i, a in enumerate(A):
    print(i)
    a = csr_matrix.tocoo(a)
    i_arr = np.repeat(i, a.row.shape)
    inds = np.stack([i_arr, a.row, a.col], axis=1)
    indices.append(inds)
    values.append(a.data)

indices = np.concatenate(indices)
values = np.concatenate(values)

# K.tf.enable_eager_execution()

A = K.tf.SparseTensor(indices, values, shape)
x = tf.convert_to_tensor(np.array(X))

A_in = Input(name='A_in', tensor=tf.cast(A, 'float32'))
X_in = Input(name='X_in', tensor=tf.cast(x, 'float32'))

h = GraphSAGELayer(aggregator=None,
                   neighbourhood_sampler=None,
                   n_neighbour_samples_per_node=2,
                   output_dim=10,
                   activation=softmax,
                   name='gs')([A_in, X_in])

reshape = keras.layers.Reshape([28*7])(h)
out = keras.layers.Dense(2, activation=keras.activations.softmax, dtype='float64')(reshape)

model = Model(inputs=[A_in, X_in], outputs=out)
model.compile(Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.summary()

x = np.array(X)
print(x.shape)
y = keras.utils.to_categorical(Y)
print(y.shape)

a = tf.data.Dataset.from_tensors(A)

model.fit(y=y, batch_size=188, epochs=100)
