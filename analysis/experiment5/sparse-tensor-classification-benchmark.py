import os
import sys
from time import time
from datetime import datetime

from keras import Model
from keras.optimizers import Adam
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

sys.path.append("/home/pete/Dropbox/PycharmProjects/graph-classifier-jan-2019")
sys.path.append("/home/pete/miniconda3/envs/graph-classifier/lib/python3.6/site-packages")
from model.GraphSAGELayer import GraphSAGELayer
import tensorflow as tf
import keras as k
import keras.backend as K
import numpy as np

from tensorflow.python import debug as tf_debug

from utils.graphs import get_data


def to_tensors(A, X, Y):
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

        a_train = to_sparse_tensor(a_train)
        a_test = to_sparse_tensor(a_test)
        x_train = tf.convert_to_tensor(np.array(x_train))
        x_test = tf.convert_to_tensor(np.array(x_test))
        y_train = tf.convert_to_tensor(k.utils.to_categorical(y_train))
        y_test = tf.convert_to_tensor(k.utils.to_categorical(y_test))
        splits.append((a_train, a_test, x_train, x_test, y_train, y_test))

    return splits


def to_sparse_tensor(A):
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
    values = K.flatten(values)
    shape = np.array(shape)
    a = tf.SparseTensor(indices, values, shape)

    return a


def build_model(a, x):
    global model
    A_in = k.models.Input(tensor=a,
                          name='Ain')
    X_in = k.models.Input(tensor=x,
                          name='Xin')
    feats = GraphSAGELayer(None, None, 2,
                           name='gs1x',
                           output_dim=32,
                           num_nodes=28)([A_in, X_in])
    feats = GraphSAGELayer(None, None, 2,
                           name='gs2x',
                           output_dim=32,
                           num_nodes=28)([A_in, feats])

    att = GraphSAGELayer(None, None, 2,
                         name='gs1a',
                         output_dim=32,
                         num_nodes=28)([A_in, X_in])
    att = GraphSAGELayer(None, None, 2,
                         name='gs2a',
                         output_dim=32,
                         num_nodes=28)([A_in, att])
    att = k.layers.Softmax(axis=1, name='softmax')(att)

    h = k.layers.Dot(axes=[1, 1])([att, feats])
    h = k.layers.Flatten()(h)
    h = k.layers.Dense(units=2,
                       activation=k.activations.softmax,
                       name='output')(h)
    model = Model(inputs=[A_in, X_in], outputs=h)
    model.compile(Adam(lr=0.01),
                  loss=k.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    return model


EAGER = False

if EAGER:
    tf.enable_eager_execution()

    all_metrics = []
    for i, split in enumerate(to_tensors(*get_data('MUTAG'))):
        print(f'\nSplit {i}')

        A_train, A_test, X_train, X_test, Y_train, Y_test = split

        model = build_model(A_train, X_train)
        model.summary(200)

        model.fit(y=Y_train,
                  epochs=80,
                  steps_per_epoch=1,
                  verbose=1)
        w = model.get_weights()

        model2 = build_model(A_test, X_test)
        model2.set_weights(w)

        loss, acc = model2.evaluate(y=Y_test, steps=1)

        preds = model2.predict(None, steps=1)
        preds = np.apply_along_axis(np.argmax, axis=1, arr=preds)

        cf = confusion_matrix(Y_test, preds)
        metrics = {'loss': loss, 'acc': acc, 'cf': cf}
        all_metrics.append(metrics)
        print(metrics)


else:
    # with tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "the-proff:9876") as s:
    with tf.Session() as s, open('out.csv', mode='a') as file:

        import csv

        fieldnames = ['time', 'trial', 'loss', 'accuracy', 'cf']
        writer = csv.DictWriter(file,
                                fieldnames=fieldnames,
                                delimiter=';')

        if not os.path.isfile('out.csv'):
            writer.writeheader()
            file.flush()

        k.backend.set_session(s)
        all_metrics = []
        for i, split in enumerate(to_tensors(*get_data('MUTAG'))):
            print(f'\nSplit {i}')

            date_time_format = "%Y-%m-%d %H:%M:%S"
            logdir = f'./logs/{datetime.now().strftime(date_time_format)}/{i}'

            A_train, A_test, X_train, X_test, Y_train, Y_test = split

            model = build_model(A_train, X_train)
            # model.summary(200)
            tb_callback = k.callbacks.TensorBoard(log_dir=logdir,
                                                  histogram_freq=0,
                                                  write_grads=False,
                                                  write_graph=True,
                                                  write_images=False)

            model.fit(y=Y_train,
                      epochs=20,
                      steps_per_epoch=1,
                      callbacks=[tb_callback],
                      verbose=1)
            w = model.get_weights()

            model2 = build_model(A_test, X_test)
            model2.set_weights(w)

            loss, acc = model2.evaluate(y=Y_test, steps=1)

            preds = model2.predict(None, steps=1)
            preds = np.apply_along_axis(np.argmax, axis=1, arr=preds)

            Y_test = s.run(Y_test)
            Y_test = np.apply_along_axis(np.argmax, axis=1, arr=Y_test)

            cf = confusion_matrix(Y_test, preds)
            metrics = {
                'time': datetime.now().strftime(date_time_format),
                'trial': i,
                'loss': loss,
                'accuracy': acc,
                'cf': cf.tolist()
            }
            all_metrics.append(metrics)
            print(metrics)
            writer.writerow(metrics)
            file.flush()

    accs = list(map(lambda x: x['accuracy'], all_metrics))
    print("mean acc:", np.mean(accs))
    print("std. dev:", np.std(accs))
