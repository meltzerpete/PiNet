import csv
from datetime import datetime
from time import time
import argparse

import keras
import keras.backend as k
import numpy as np
import tensorflow as tf
from keras import Model
from keras.optimizers import Adam
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from model.GraphSAGELayer import GraphSAGELayer
from utils import get_data


class PiNetGraphSage:

    def __init__(self, aggregator='mean', epochs=50, num_samples_per_node=2):
        self.epochs = epochs
        self.agg = aggregator
        self.num_samples_per_node = num_samples_per_node

    def to_tensors(self, A, X, Y):
        Y = Y['graph_label'].apply(lambda x: 1 if x == 1 else 0)

        A = np.array(A)
        X = np.array(X)
        # A_train, A_test, X_train, X_test, Y_train, Y_test = \
        #     train_test_split(A, X, Y, test_size=0.1)

        splits = []
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        for train_index, test_index in skf.split(X, Y):
            a_train, a_test = A[train_index], A[test_index]
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            a_train = self.to_sparse_tensor(a_train)
            a_test = self.to_sparse_tensor(a_test)
            x_train = tf.convert_to_tensor(np.array(x_train))
            x_test = tf.convert_to_tensor(np.array(x_test))
            y_train = tf.convert_to_tensor(keras.utils.to_categorical(y_train))
            y_test = tf.convert_to_tensor(keras.utils.to_categorical(y_test))
            splits.append((a_train, a_test, x_train, x_test, y_train, y_test))

        return splits

    def to_sparse_tensor(self, A):
        shape = (len(A), A[0].shape[-1], A[0].shape[-1])
        indices = []
        values = []
        for i, a in enumerate(A):
            a = csr_matrix.tocoo(a)
            i_arr = np.repeat(np.array(i, dtype='int64'), a.row.shape)
            inds = np.stack([i_arr, a.row, a.col], axis=1)
            indices.append(inds)
            values.append(a.data)

        indices = tf.convert_to_tensor(np.concatenate(indices), name="indices")
        values = tf.convert_to_tensor(np.concatenate(values))
        values = k.flatten(values)
        shape = np.array(shape)
        a = tf.SparseTensor(indices, values, shape)

        return a

    def build_model(self, a, x, aggregator='max', n_classes=2):
        A_in = keras.models.Input(tensor=a,
                                  name='Ain')
        X_in = keras.models.Input(tensor=x,
                                  name='Xin')
        s = k.get_session()
        X = s.run(x)
        num_nodes = X.shape[-2]

        feats = GraphSAGELayer(None, 2,
                               name='gs1x',
                               output_dim=32,
                               aggregator=aggregator,
                               aggregator_dims=20,
                               num_nodes=num_nodes)([A_in, X_in])
        feats = GraphSAGELayer(None, 2,
                               name='gs2x',
                               aggregator=aggregator,
                               aggregator_dims=20,
                               output_dim=32,
                               num_nodes=num_nodes)([A_in, feats])

        att = GraphSAGELayer(None, 2,
                             name='gs1a',
                             aggregator=aggregator,
                             aggregator_dims=20,
                             output_dim=32,
                             num_nodes=num_nodes)([A_in, X_in])
        att = GraphSAGELayer(None, 2,
                             name='gs2a',
                             aggregator=aggregator,
                             aggregator_dims=20,
                             output_dim=32,
                             num_nodes=num_nodes)([A_in, att])
        att = keras.layers.Softmax(axis=1, name='softmax')(att)

        # att = keras.layers.Dropout(0.5)(att)
        # feats = keras.layers.Dropout(0.5)(feats)

        h = keras.layers.Dot(axes=[1, 1])([att, feats])
        h = keras.layers.Flatten()(h)
        h = keras.layers.Dense(units=n_classes,
                               activation=keras.activations.softmax,
                               name='output')(h)
        model = Model(inputs=[A_in, X_in], outputs=h)
        model.summary()
        model.compile(Adam(lr=0.01),
                      loss=keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])

        return model

    def get_accs_times(self, A, X, Y, classes, splits=None, batch_size=None):

        fieldnames = ['time_stamp', 'aggregator', 'trial', 'loss', 'accuracy', 'train_time', 'cf']
        with open('PiNet-out.csv', 'a') as file:
            writer = csv.DictWriter(file,
                                    fieldnames=fieldnames,
                                    delimiter=';')

            writer.writeheader()
            file.flush()

            # args = {'jobs': }
            # config = tf.ConfigProto(intra_op_parallelism_threads=args['jobs'],
            #                         inter_op_parallelism_threads=args['jobs'],
            #                         allow_soft_placement=True,
            #                         device_count={'CPU': args['jobs']})

            # config = tf.ConfigProto()
            # config.log_device_placement = True
            # config.gpu_options.allow_growth = True
            # session = tf.Session(config=config)
            # k.set_session(session)

            all_metrics = []
            for i, split in enumerate(self.to_tensors(A, X, Y)):
                print(f'\nPiNetGraphSAGE\nSplit {i}')

                start = time()
                date_time_format = "%Y-%m-%d %H:%M:%S"
                # logdir = f'./logs/{datetime.now().strftime(date_time_format)}/{i}'

                A_train, A_test, X_train, X_test, Y_train, Y_test = split

                model = self.build_model(A_train, X_train, aggregator=self.agg, n_classes=classes)
                # model.summary(200)
                # tb_callback = keras.callbacks.TensorBoard(log_dir=logdir,
                #                                           histogram_freq=0,
                #                                           write_grads=False,
                #                                           write_graph=True,
                #                                           write_images=False)

                model.fit(y=Y_train,
                          epochs=self.epochs,
                          steps_per_epoch=1,
                          # callbacks=[tb_callback],
                          verbose=1)

                finish = time()

                w = model.get_weights()

                model2 = self.build_model(A_test, X_test, aggregator=self.agg, n_classes=classes)
                model2.set_weights(w)

                loss, acc = model2.evaluate(y=Y_test, steps=1)

                preds = model2.predict(None, steps=1)
                preds = np.apply_along_axis(np.argmax, axis=1, arr=preds)

                s = k.get_session()
                Y_test = s.run(Y_test)
                Y_test = np.apply_along_axis(np.argmax, axis=1, arr=Y_test)

                cf = confusion_matrix(Y_test, preds)
                metrics = {
                    'time_stamp': datetime.now().strftime(date_time_format),
                    'aggregator': self.agg,
                    'trial': i,
                    'loss': loss,
                    'accuracy': acc,
                    'train_time': finish - start,
                    'cf': cf.tolist()
                }
                all_metrics.append(metrics)
                print(metrics)
                writer.writerow(metrics)
                file.flush()

        # k.clear_session()
        accs, times = zip(*list(map(lambda x: (x['accuracy'], x['train_time']), all_metrics)))

        print("mean acc:", np.mean(accs))
        print("std. dev:", np.std(accs))

        return accs, times

    def name(self):
        return f'PiNet-GraphSAGE-{self.agg}'
