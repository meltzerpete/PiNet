from math import ceil

import numpy as np
from keras import Model
from keras import backend as K
from keras.layers import Dense, Flatten, Lambda
from keras.models import Input
from keras.optimizers import Adam

from model.ClassificationAccuracyTimeBenchmark import split_test_train
from model.GraphClassifier import GraphClassifier
from model.MyGCN import MyGCN


class GCNwithOptionalSum:
    def __init__(self, with_sum=False):
        self.with_sum = with_sum

    def name(self):
        return 'GCNWithSum' if self.with_sum else 'GCNNoSum'

    def get_accuracies(self, A, X, y, num_graph_classes, splits=None, batch_size=None):
        accuracies = []
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
            model.fit_generator(generator, ceil(y_train.shape[0] / batch_size), 200, verbose=0)

            stats = model.evaluate_generator(
                GraphClassifier().batch_generator([A_test, X_test], y_test, batch_size), y_test.shape[0] / batch_size)

            # for metric, val in zip(model.metrics_names, stats):
            #     print(metric + ": ", val)

            accuracies.append(stats[1])

        # print("mean acc:", np.mean(accuracies))
        # print("std:", np.std(accuracies))
        return accuracies
