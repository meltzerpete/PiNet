from math import ceil

import numpy as np
from keras import Model
from keras.layers import Dense, Flatten
from keras.models import Input
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit

from analysis.experiment2 import generate
from model.ClassificationAccuracyTimeBenchmark import split_test_train
from model.GraphClassifier import GraphClassifier
from model.MyGCN import MyGCN

num_nodes_per_graph=50
num_graph_classes=5
num_node_classes=2
num_graphs_per_class=100
batch_size = 5

A, X, y = generate.get_tensors(num_nodes_per_graph,
                               num_graph_classes,
                               num_node_classes,
                               num_graphs_per_class)


# splits = StratifiedKFold(10, shuffle=True).split(X, y)
splits = StratifiedShuffleSplit(n_splits=10, train_size=num_graph_classes*2).split(X, y)

accuracies = []
for train_idx, val_idx in iter(splits):

    A_test, A_train, X_test, X_train, y_test, y_train \
        = split_test_train(A, X, y, train_idx, val_idx)
    print(A_train.shape, X_train.shape, y_train.shape)

    A_in = Input((A[0].shape[0], A[0].shape[1]), name='A_in')
    X_in = Input(X[0].shape, name='X_in')

    x1 = MyGCN(100, activation='relu')([A_in, X_in])
    x2 = MyGCN(64, activation='relu')([A_in, x1])
    x3 = Flatten()(x2)
    x4 = Dense(num_graph_classes, activation='softmax')(x3)

    model = Model(inputs=[A_in, X_in], outputs=x4)

    # print(model.summary())

    model.compile(Adam(), loss='categorical_crossentropy', metrics=['acc'])
    generator = GraphClassifier().batch_generator([A_train, X_train], y_train, batch_size)
    model.fit_generator(generator, ceil(y_train.shape[0] / batch_size), 200, verbose=0)

    stats = model.evaluate_generator(
        GraphClassifier().batch_generator([A_test, X_test], y_test, 5), y_test.shape[0] / batch_size)

    for metric, val in zip(model.metrics_names, stats):
        print(metric + ": ", val)

    accuracies.append(stats[1])

print("mean acc:", np.mean(accuracies))
print("std:", np.std(accuracies))