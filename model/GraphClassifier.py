import time
from math import ceil

import keras
import numpy as np
from keras.layers import Input, Dot, Reshape, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from scipy.sparse import csr_matrix

from model.MyGCN import MyGCN


class GraphClassifier:
    """
    Graph Classifier
        :param A: Adjacency matrices - List of ndarrays
        :param X: Features matrices - List of ndarrays
        :param Y: Labels - (n x 1) ndarray
        :param dataset_name: String
        :param num_classes: default 2
        :param out_dim_a2: output dimension for attention
        :param out_dim_x2: output dimension for features
        :param tensor_board_logging: enable logginf for TensorBoard
        :param reduce_lr_callback: reduce learning rate based on validation set
        """
    def __init__(self, A, X, Y, dataset_name='dataset', num_classes=2,
                 out_dim_a2=64, out_dim_x2=64, tensor_board_logging=False,
                 reduce_lr_callback=False):
        self._A = A
        self._X = X
        self._Y = Y
        self._dataset_name = dataset_name
        self._num_classes = num_classes
        self._out_dim_a2 = out_dim_a2
        self._out_dim_x2 = out_dim_x2
        self._tensor_board_logging = tensor_board_logging
        self._reduce_lr_callback = reduce_lr_callback

        self._model = None
        self._history = None

    def preprocess_A(self, preprocess_A=None):
        if preprocess_A is None:
            preprocess_A = []
        A = self._A.copy()

        if 'add_self_loops' in preprocess_A:
            A = map(self._add_self_loops, A)

        if 'sym_normalise_A' in preprocess_A:
            A = map(self._sym_normalise_A, A)

        if 'laplacian' in preprocess_A:
            A = map(self._laplacian, A)

        if 'sym_norm_laplacian' in preprocess_A:
            A = map(self._sym_norm_laplacian, A)

        self._A = list(map(csr_matrix, A))

    def get_predictions(self, A, X, Y, batch_size=50):
        steps = ceil(Y.shape[0] / batch_size)
        return self._model.predict_generator(
            generator=self._pred_batch_generator([A, X], Y, batch_size), steps=steps)

    def build_fit_eval(self, epochs=200, batch_size=50, folds=None, verbose=1):
        accuracies = []
        times = []
        for j, (train_idx, val_idx) in enumerate(folds):
            print("split :", j)

            A_test, A_train, X_test, X_train, Y_test, Y_train \
                = self._split_test_train(train_idx, val_idx)

            self._model = self._define_model()
            self._model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
            # model.summary()

            callbacks = []

            if self._tensor_board_logging:
                tb_callback = keras.callbacks.TensorBoard(log_dir='./Graph/' + self._dataset_name + '/' + str(j),
                                                          histogram_freq=0,
                                                          write_grads=False,
                                                          write_graph=True, write_images=False)
                callbacks.append(tb_callback)

            if self._reduce_lr_callback:
                reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                       factor=0.1,
                                                                       patience=3,
                                                                       verbose=verbose)
                callbacks.append(reduce_lr_callback)

            steps = ceil(Y_test.shape[0] / batch_size)

            start = time.time()
            self._history = self._model.fit_generator(
                generator=self._batch_generator([A_train, X_train], Y_train, batch_size),
                epochs=epochs,
                steps_per_epoch=steps,
                callbacks=callbacks,
                verbose=verbose)

            train_time = time.time() - start

            print("train time: ", train_time)
            times.append(train_time)

            stats = self._model.evaluate_generator(
                generator=self._batch_generator([A_test, X_test], Y_test, batch_size),
                steps=steps)

            for metric, val in zip(self._model.metrics_names, stats):
                print(metric + ": ", val)

            accuracies.append(stats[1])
        return accuracies, times

    def _add_self_loops(self, A):
        np.fill_diagonal(A, 1)
        return A

    def _sym_normalise_A(self, A):
        # may produce / by 0 warning but this is ok: inf replaced by 0
        d_hat_with_inf = 1 / np.sqrt(np.sum(A, axis=1))
        d_hat_with_inf[np.isinf(d_hat_with_inf)] = 0
        D_hat = np.diag(d_hat_with_inf)
        A_ = np.dot(D_hat, A).dot(D_hat)
        return A_

    def _laplacian(self, A):
        D = np.diag(np.sum(A, 0))
        return D - A

    def _sym_norm_laplacian(self, A):
        I = np.eye(np.shape(A)[0])
        return I - self._sym_normalise_A(A)

    def _define_model(self):
        A_in = Input((self._X[0].shape[0], self._X[0].shape[0]), name='A_in')
        X_in = Input(self._X[0].shape, name='X_in')

        x1 = MyGCN(100, activation='relu')([A_in, X_in])
        x1 = MyGCN(self._out_dim_a2, activation='softmax')([A_in, x1])

        x2 = MyGCN(100, activation='relu')([A_in, X_in])
        x2 = MyGCN(self._out_dim_x2, activation='relu')([A_in, x2])

        x3 = Dot(axes=[1, 1])([x1, x2])
        x3 = Reshape((self._out_dim_a2 * self._out_dim_x2,))(x3)
        x4 = Dense(self._num_classes, activation='softmax')(x3)

        return Model(inputs=[A_in, X_in], outputs=x4)

    def _split_test_train(self, train_idx, val_idx):
        A_test = np.array([self._A[i] for i in val_idx])
        A_train = np.array([self._A[i] for i in train_idx])

        X_test = np.array([self._X[i] for i in val_idx])
        X_train = np.array([self._X[i] for i in train_idx])

        Y_test = to_categorical(self._Y)[val_idx]
        Y_train = to_categorical(self._Y)[train_idx]

        return A_test, A_train, X_test, X_train, Y_test, Y_train

    def _batch_generator(self, A_X, y, batch_size):
        number_of_batches = ceil(y.shape[0] / batch_size)
        counter = 0
        shuffle_index = np.arange(np.shape(y)[0])
        np.random.shuffle(shuffle_index)
        A_ = A_X[0]
        X = A_X[1]
        A_ = A_[shuffle_index]
        X = X[shuffle_index]
        y = y[shuffle_index]
        while 1:
            index_batch = shuffle_index[batch_size * counter:min(batch_size * (counter + 1), y.shape[0])]
            if len(A_.shape) == 1:
                A_batch = np.array(list(map(lambda a: csr_matrix.todense(a), A_[index_batch].tolist())))
            else:
                A_batch = A_[index_batch]
            X_batch = X[index_batch]
            y_batch = y[index_batch]
            counter += 1
            yield ([A_batch, X_batch], y_batch)
            if (counter < number_of_batches):
                np.random.shuffle(shuffle_index)
                counter = 0

    def _pred_batch_generator(self, A_X, y, batch_size):
        # number_of_batches = ceil(y.shape[0] / batch_size)
        counter = 0
        indexes = np.arange(np.shape(y)[0])

        A_ = A_X[0]
        X = A_X[1]

        while 1:
            batch_indexes = indexes[batch_size * counter:min(batch_size * (counter + 1), y.shape[0])]
            A_batch = A_[batch_indexes]
            X_batch = X[batch_indexes]
            y_batch = y[batch_indexes]
            yield ([A_batch, X_batch], y_batch)
            counter += 1
