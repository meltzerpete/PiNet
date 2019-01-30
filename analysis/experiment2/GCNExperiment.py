import numpy as np
from keras import Model
from keras.layers import Dense, Flatten
from keras.models import Input
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold

from analysis.experiment2 import generate
from model.ClassificationAccuracyTimeBenchmark import split_test_train
from model.GraphClassifier import GraphClassifier
from model.MyGCN import MyGCN

num_nodes_per_graph=100
num_graph_classes=10
num_node_classes=2
num_graphs_per_class=50

A, X, y = generate.get_tensors(num_nodes_per_graph,
                               num_graph_classes,
                               num_node_classes,
                               num_graphs_per_class)


splits = StratifiedKFold(10, shuffle=True).split(X, y)

# y = to_categorical(Y, num_graph_classes)
accuracies = []
for train_idx, val_idx in iter(splits):

    A_test, A_train, X_test, X_train, y_test, y_train \
        = split_test_train(A, X, y, train_idx, val_idx)

    A_in = Input((A[0].shape[0], A[0].shape[1]), name='A_in')
    X_in = Input(X[0].shape, name='X_in')

    x1 = MyGCN(100, activation='relu')([A_in, X_in])
    x2 = MyGCN(64, activation='relu')([A_in, x1])
    x3 = Flatten()(x2)
    x4 = Dense(num_graph_classes, activation='softmax')(x3)

    model = Model(inputs=[A_in, X_in], outputs=x4)

    print(model.summary())

    model.compile(Adam(), loss='categorical_crossentropy', metrics=['acc'])
    generator = GraphClassifier().batch_generator([A_train, X_train], y_train, 5)
    model.fit_generator(generator, X.shape[0] / 5, 20)

    stats = model.evaluate_generator(GraphClassifier().batch_generator([A_test, X_test], y_test, 5), X.shape[0] / 5)

    for metric, val in zip(model.metrics_names, stats):
        print(metric + ": ", val)

    accuracies.append(stats[1])

print("mean acc:", np.mean(accuracies))
print("std:", np.std(accuracies))