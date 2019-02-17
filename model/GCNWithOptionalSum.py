from math import ceil

import numpy as np
from keras import Model
from keras import backend as K
from keras.layers import Dense, Flatten, Lambda
from keras.models import Input
from keras.optimizers import Adam
from scipy.sparse import csr_matrix

from model.ClassificationAccuracyTimeBenchmark import split_test_train
from model.GraphClassifier import GraphClassifier
from model.MyGCN import MyGCN

import time


class GCNWithOptionalSum:
    def __init__(self, with_sum=False):
        self.with_sum = with_sum

    def name(self):
        return 'GCNWithSum' if self.with_sum else 'GCNNoSum'

    def get_accuracies(self, A, X, Y, num_graph_classes, splits=None, batch_size=None):
        return self.get_accs_times(A, X, Y, num_graph_classes, splits, batch_size)[0]

    def get_accs_times(self, A, X, y, num_graph_classes, splits=None, batch_size=50):

        A = map(csr_matrix.todense, A)
        A = map(self._add_self_loops, A)
        A = map(self._sym_normalise_A, A)
        A = list(map(csr_matrix, A))

        accuracies = []
        times = []
        for train_idx, val_idx in iter(splits):
            A_test, A_train, X_test, X_train, y_test, y_train \
                = split_test_train(A, X, y, train_idx, val_idx)

            A_in = Input((A[0].shape[0], A[0].shape[1]), name='A_in')
            X_in = Input(X[0].shape, name='X_in')

            x1 = MyGCN(100, activation='relu')([A_in, X_in])
            x2 = MyGCN(64, activation='relu')([A_in, x1])
            x3 = Lambda(lambda x: K.sum(x, axis=1))(x2) if self.with_sum else Flatten()(x2)
            x4 = Dense(num_graph_classes, activation='softmax')(x3)

            model = Model(inputs=[A_in, X_in], outputs=x4)

            # print(model.summary())

            model.compile(Adam(), loss='categorical_crossentropy', metrics=['acc'])
            generator = GraphClassifier().batch_generator([A_train, X_train], y_train, batch_size)
            start = time.time()
            model.fit_generator(generator, ceil(y_train.shape[0] / batch_size), 200, verbose=0)
            train_time = time.time() - start

            stats = model.evaluate_generator(
                GraphClassifier().batch_generator([A_test, X_test], y_test, batch_size), y_test.shape[0] / batch_size)

            # for metric, val in zip(model.metrics_names, stats):
            #     print(metric + ": ", val)

            accuracies.append(stats[1])
            times.append(train_time)

        # print("mean acc:", np.mean(accuracies))
        # print("std:", np.std(accuracies))
        return accuracies, times

    def _add_self_loops(self, A):
        np.fill_diagonal(A, 1)
        return A

    def _sym_normalise_A(self, A):
        # may produce / by 0 warning but this is ok: inf replaced by 0
        d_hat_with_inf = 1 / np.sqrt(np.sum(A, axis=1))
        d_hat_with_inf[np.isinf(d_hat_with_inf)] = 0
        D_hat = np.diagflat(d_hat_with_inf)
        A_ = np.dot(D_hat, A).dot(D_hat)
        return A_
