from keras.models import Model
from math import ceil
from keras.utils import to_categorical
from ImportData import DropboxLoader
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csr_matrix
from keras.layers import Input, Dot, Reshape, Dense, Dropout
from keras.optimizers import Adam
from model.MyGCN import MyGCN
import keras
import pickle
import time
from csv import writer
import os


def preprocess(A):
    np.fill_diagonal(A, 1)
    A_hat = A
    D_hat = np.diag(1 / np.sqrt(np.sum(A, axis=1)))
    A_ = np.dot(A_hat, D_hat).dot(A_hat)
    return A_


def get_A_X(loader):
    all_adj = loader.get_adj()
    graph_ind = loader.get_graph_ind()
    numGraphs = graph_ind.max()['graph_ind']

    def get_A_X_by_graph_id(id):
        nodes_to_keep = graph_ind.loc[graph_ind['graph_ind'] == id]['node']
        adj = all_adj[all_adj['from'].isin(nodes_to_keep) | all_adj['to'].isin(nodes_to_keep)]

        all_X = loader.get_node_label()
        ids = all_X[all_X['node'].isin(nodes_to_keep)].index
        X = pd.get_dummies(all_X['label']).iloc[ids].values
        G = nx.from_pandas_edgelist(adj, 'from', 'to')
        A = nx.to_numpy_array(G)
        return A, X

    A, X = zip(*[get_A_X_by_graph_id(id) for id in range(1, numGraphs + 1)])
    num_nodes_list = list(map(lambda a: a.shape[0], X))
    maxNodes = max(num_nodes_list)

    def normaliseX_by_num_nodes(x_num_nodes):
        x, num_nodes = x_num_nodes
        return np.divide(x, num_nodes)

    def padA(a):
        padA = np.zeros([maxNodes, maxNodes])
        padA[:a.shape[0], :a.shape[1]] = a
        return padA

    def padX(x):
        padX = np.zeros([maxNodes, x.shape[1]])
        padX[:x.shape[0], :x.shape[1]] = x
        return padX

    padA = list(map(padA, A))
    padX = list(map(padX, map(normaliseX_by_num_nodes, zip(X, num_nodes_list))))

    return padA, padX


def get_data(dropbox_name, file_name_ext):
    dataset = DropboxLoader(dropbox_name)

    if os.path.isfile(dropbox_name + file_name_ext):
        A_list, X = pickle.load(open(dropbox_name + file_name_ext, "rb"))
    else:
        A, X = get_A_X(dataset)
        A_list = list(map(csr_matrix, map(preprocess, A)))
        pickle.dump((A_list, X), open(dropbox_name + file_name_ext, "wb"))
    Y = dataset.get_graph_label()

    return A_list, X, Y


def main():
    with open('out.csv', 'a') as csv_file:
        res_writer = writer(csv_file, delimiter=';')
        res_writer.writerow(
            ["dataset", "batch_size", "mean_acc", "acc_std", "mean_train_time(s)", "time_std", "all_accs", "all_times"])

        datasets = {
            'ENZYMES': {
                'preprocess_graph_labels': lambda x: x - 1,
                'classes': 6,
            },
            'MUTAG': {

            },
            'NCI1': {
                'pretty_name': 'NCI-1',
            },
            'NCI109': {
                'pretty_name': 'NCI-109',
            },
            'PTC_MM': {
                'pretty_name': 'PTC-MM',
            },
            'PTC_FM': {
                'pretty_name': 'PTC-FM',
            },
            'PTC_MR': {
                'pretty_name': 'PTC-MR',
            },
            'PTC_FR': {
                'pretty_name': 'PTC-FR',
            },
        }

        batch_sizes = [50]
        file_name_ext = "_normalise.p"

        out_dim_a2 = 64
        out_dim_x2 = 64

        for dataset_name, dataset in datasets.items():

            print("Dataset: ", dataset_name)

            dropbox_name = dataset['dropbox_name'] \
                if 'dropbox_name' in dataset.keys() else dataset_name

            preprocess_graph_labels = dataset['preprocess_graph_labels'] \
                if 'preprocess_graph_labels' in dataset.keys() else lambda x: 1 if x == 1 else 0

            classes = dataset['classes'] \
                if 'classes' in dataset.keys() else 2

            for batch_size in batch_sizes:
                # prepare data
                print("preparing data")

                A_list, X, Y = get_data(dropbox_name, file_name_ext)

                Y['graph_label'] = Y['graph_label'].apply(preprocess_graph_labels)
                Y = np.array(Y)

                folds = list(StratifiedKFold(n_splits=10, shuffle=True).split(X, Y))

                accuracies = []
                times = []
                for j, (train_idx, val_idx) in enumerate(folds):
                    print("split :", j)

                    A_test, A_train, X_test, X_train, Y_test, Y_train \
                        = split_test_train(A_list, X, Y, train_idx, val_idx)

                    model = define_model(X, classes, out_dim_a2, out_dim_x2)

                    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

                    model.summary()

                    tb_callback = keras.callbacks.TensorBoard(log_dir='./Graph/' + dropbox_name + '/' + str(j),
                                                              histogram_freq=0,
                                                              write_grads=False,
                                                              write_graph=True, write_images=False)

                    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                           factor=0.1,
                                                                           patience=3,
                                                                           verbose=1)

                    steps = ceil(Y_test.shape[0] / batch_size)

                    start = time.time()
                    history = model.fit_generator(generator=batch_generator([A_train, X_train], Y_train, batch_size),
                                                  epochs=200,
                                                  steps_per_epoch=steps,
                                                  # callbacks=[tb_callback],
                                                  verbose=1)

                    train_time = time.time() - start

                    print("train time: ", train_time)
                    times.append(train_time)

                    stats = model.evaluate_generator(generator=batch_generator([A_test, X_test], Y_test, batch_size),
                                                     steps=steps)

                    for metric, val in zip(model.metrics_names, stats):
                        print(metric + ": ", val)

                    accuracies.append(stats[1])

                print(dropbox_name)
                mean_acc = np.mean(accuracies)
                print("mean acc:", mean_acc)
                acc_std = np.std(accuracies)
                print("std dev:", acc_std)
                print("accs:", accuracies)
                mean_time = np.mean(times)
                print("mean train time", mean_time)
                time_std = np.std(times)
                print("std dev:", time_std, end="\n\n")

                res_writer.writerow(
                    [dropbox_name, batch_size, mean_acc, acc_std, mean_time, time_std, accuracies, times])
                csv_file.flush()


def define_model(X, classes, out_dim_a2, out_dim_x2):
    A_in = Input((X[0].shape[0], X[0].shape[0]), name='A_in')
    X_in = Input(X[0].shape, name='X_in')

    x1 = MyGCN(100, activation='relu')([A_in, X_in])
    x1 = MyGCN(out_dim_a2, activation='softmax')([A_in, x1])

    x2 = MyGCN(100, activation='relu')([A_in, X_in])
    x2 = MyGCN(out_dim_x2, activation='relu')([A_in, x2])

    x3 = Dot(axes=[1, 1])([x1, x2])
    x3 = Reshape((out_dim_a2 * out_dim_x2,))(x3)
    x4 = Dense(classes, activation='softmax')(x3)

    return Model(inputs=[A_in, X_in], outputs=x4)


def split_test_train(A_list, X, Y, train_idx, val_idx):
    A_test = np.array([A_list[i] for i in val_idx])
    A_train = np.array([A_list[i] for i in train_idx])

    X_test = np.array([X[i] for i in val_idx])
    X_train = np.array([X[i] for i in train_idx])

    Y_test = to_categorical(Y)[val_idx]
    Y_train = to_categorical(Y)[train_idx]

    return A_test, A_train, X_test, X_train, Y_test, Y_train


def batch_generator(A_X, y, batch_size):
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
        index_batch = shuffle_index[
                      batch_size * counter:min(batch_size * (counter + 1), y.shape[0])]
        A_batch = np.array(list(map(lambda a: csr_matrix.todense(a), A_[index_batch].tolist())))
        X_batch = X[index_batch]
        y_batch = y[index_batch]
        counter += 1
        yield ([A_batch, X_batch], y_batch)
        if (counter < number_of_batches):
            np.random.shuffle(shuffle_index)
            counter = 0

main()