import numpy as np

from model.GraphSAGELayer import GraphSAGELayer
from utils.graphs import get_data

import keras
import keras.backend as K
import numpy as np
from keras.activations import softmax
from keras.layers import Input, Dot, Reshape, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam

A, X, Y = get_data('MUTAG')

X = np.array(X)

A_in = Input(A[0].shape, name='A_in')
X_in = Input(X[0].shape, name='X_in')

h = GraphSAGELayer(None, None, 10, activation=softmax)([A_in, X_in])

model = Model(inputs=[A_in, X_in], outputs=h)

# model.compile(Adam)

model.summary()

# x1 = MyGCN(100, activation='relu', learn_pqr=True)([A_in, X_in])
# x1 = MyGCN(self._out_dim_a2, activation='relu', learn_pqr=True)([A_in, x1])
# x1 = Lambda(lambda X: K.transpose(softmax(K.transpose(X))))(x1)
#
# x2 = MyGCN(100, activation='relu', learn_pqr=True)([A_in, X_in])
# x2 = MyGCN(self._out_dim_x2, activation='relu', learn_pqr=True)([A_in, x2])
#
# x3 = Dot(axes=[1, 1])([x1, x2])
# x3 = Reshape((self._out_dim_a2 * self._out_dim_x2,))(x3)
# x4 = Dense(num_classes, activation='softmax')(x3)
#
