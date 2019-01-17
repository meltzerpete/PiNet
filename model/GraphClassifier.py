from keras.models import Model
from math import ceil
from keras.utils import to_categorical
from ImportData import DropboxLoader as dl
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

csv_file = open('out.csv', 'a')
res_writer = writer(csv_file, delimiter=';')
res_writer.writerow(
    ["dataset", "batch_size", "mean_acc", "acc_std", "mean_train_time(s)", "time_std", "all_accs", "all_times"])


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
    maxNodes = max(map(lambda a: a.shape[0], A))

    def padA(a):
        padA = np.zeros([maxNodes, maxNodes])
        padA[:a.shape[0], :a.shape[1]] = a
        return padA

    def padX(x):
        padX = np.zeros([maxNodes, x.shape[1]])
        padX[:x.shape[0], :x.shape[1]] = x
        return padX

    padA = list(map(padA, A))
    padX = list(map(padX, X))

    # stackA = np.stack(padA)
    # stackX = np.stack(padX)

    return padA, padX


# for dataset_name in ['PTC_MM', 'PTC_FM', 'PTC_MR', 'PTC_FR']:
for dataset_name in ['DD']:
    # prepare data
    print("preparing data")
    dataset = dl.DropboxLoader(dataset_name)

    # A, X = get_A_X(dataset)
    # A_list = list(map(csr_matrix, A))
    # pickle.dump((A_list, X), open(dataset_name + ".p", "wb"))
    A_list, X = pickle.load(open(dataset_name + ".p", "rb"))

    Y = dataset.get_graph_label()

    Y['graph_label'] = Y['graph_label'].apply(lambda x: 1 if x == 1 else 0)
    Y = np.array(Y)

    folds = list(StratifiedKFold(n_splits=10, shuffle=True).split(X, Y))

    batch_size = 1

    accuracies = []
    times = []
    for j, (train_idx, val_idx) in enumerate(folds):
        print("split :", j)

        A_train = np.array([A_list[i] for i in train_idx])
        X_train = np.array([X[i] for i in train_idx])
        # X_train = X[train_idx]
        Y_train = to_categorical(Y)[train_idx]

        A_test = np.array([A_list[i] for i in val_idx])
        X_test = np.array([X[i] for i in val_idx])
        # X_test = X[val_idx]
        Y_test = to_categorical(Y)[val_idx]

        A_in = Input((X[0].shape[0], X[0].shape[0]), name='A_in')
        X_in = Input(X[0].shape, name='X_in')

        # x1 = Dropout(0.5)(inputs)
        x1 = MyGCN(64, activation='relu')([A_in, X_in])
        # x1 = Dropout(0.5)(x1)
        x1 = MyGCN(10, activation='softmax')([A_in, x1])
        # x1 = Dropout(0.5)(x1)

        # x2 = Dropout(0.5)(inputs)
        x2 = MyGCN(64, activation='relu')([A_in, X_in])
        # x2 = Dropout(0.5)(x2)
        x2 = MyGCN(10, activation='relu')([A_in, x2])
        # x2 = Dropout(0.5)(x2)

        x3 = Dot(axes=[1, 1])([x1, x2])
        x3 = Reshape((100,))(x3)

        x4 = Dense(2, activation='softmax')(x3)

        model = Model(inputs=[A_in, X_in], outputs=x4)
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


        # model.summary()

        # tb_callback = keras.callbacks.TensorBoard(log_dir='./Graph/' + dataset_name + '/' + str(j), histogram_freq=0,
        #                                           write_grads=False,
        #                                           write_graph=True, write_images=False)

        # reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
        #                                                        factor=0.1,
        #                                                        patience=3,
        #                                                        verbose=1)

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
                index_batch = shuffle_index[batch_size * counter:min(batch_size * (counter + 1), y.shape[0])]
                A_batch = np.array(list(map(lambda a: csr_matrix.todense(a), A_[index_batch].tolist())))
                X_batch = X[index_batch]
                y_batch = y[index_batch]
                counter += 1
                yield ([A_batch, X_batch], y_batch)
                if (counter < number_of_batches):
                    np.random.shuffle(shuffle_index)
                    counter = 0


        steps = ceil(Y_test.shape[0] / batch_size)

        start = time.time()
        history = model.fit_generator(generator=batch_generator([A_train, X_train], Y_train, batch_size),
                                      epochs=200,
                                      steps_per_epoch=steps,
                                      verbose=1)

        train_time = time.time() - start

        print("train time: ", train_time)
        times.append(train_time)

        stats = model.evaluate_generator(generator=batch_generator([A_test, X_test], Y_test, batch_size), steps=steps)

        for metric, val in zip(model.metrics_names, stats):
            print(metric + ": ", val)

        accuracies.append(stats[1])

    print(dataset_name)
    mean_acc = np.mean(accuracies)
    print("mean acc:", mean_acc)
    acc_std = np.std(accuracies)
    print("std dev:", acc_std)
    print("accs:", accuracies)
    mean_time = np.mean(times)
    print("mean train time", mean_time)
    time_std = np.std(times)
    print("std dev:", time_std, end="\n\n")

    res_writer.writerow([dataset_name, batch_size, mean_acc, acc_std, mean_time, time_std, accuracies, times])
    csv_file.flush()

csv_file.close()
