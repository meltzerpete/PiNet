from keras.models import Model
from keras import backend as K
from keras.utils import to_categorical
from ImportData import DropboxLoader as dl
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from scipy.sparse import csr_matrix
from keras.layers import Input, Lambda, Dot, Permute, Activation, Reshape, Dense, Dropout, TimeDistributed
from keras.optimizers import Adam
from keras.regularizers import l2
from model.MyGCN import MyGCN
from Models.GCN.gcn_utils import preprocess_adj, accuracy
import keras
import tensorflow as tf
import pickle


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

    stackA = np.stack(padA)
    stackX = np.stack(padX)

    return stackA, stackX


def get_G(A, X):
    # calculate A_ (symmetrically normalised adj) - existing code (preprocess_adj) requires sparse
    A_sparse = list(map(csr_matrix, A))
    A_ = np.array(list(map(lambda a: preprocess_adj(a).todense(), A_sparse)))

    # Normalize X
    X /= X.sum(2).reshape(X.shape[0], X.shape[1], 1).repeat(X.shape[2], 2)
    np.nan_to_num(X, False)

    return np.concatenate([A_, X], axis=2)


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


# todo
# l = list(map(lambda a: convert_sparse_matrix_to_sparse_tensor(csr_matrix(a)), A))

for dataset_name in ['MUTAG']:
    # prepare data
    print("preparing data")
    dataset = dl.DropboxLoader(dataset_name)
    A, X = get_A_X(dataset)
    A_sparse = list(map(csr_matrix, A))
    A_ = np.array(list(map(lambda a: preprocess_adj(a).todense(), A_sparse)))
    # G = get_G(A, X)
    # pickle.dump(G, open(dataset + ".p", "wb"))

    G = pickle.load(open(dataset_name + ".p", "rb"))
    Y = dataset.get_graph_label()

    Y['graph_label'] = Y['graph_label'].apply(lambda x: 1 if x == 1 else 0)
    Y = np.array(Y)

    folds = list(StratifiedKFold(n_splits=10, shuffle=True).split(G, Y))

    accuracies = []
    for j, (train_idx, val_idx) in enumerate(folds):
        print("split :", j)
        G_train = G[train_idx]
        A_train = A_[train_idx]
        X_train = X[train_idx]
        Y_train = to_categorical(Y)[train_idx]
        print("Y_train shape:", Y_train.shape, "Num 1s: ", Y_train.sum(0))
        G_test = G[val_idx]
        A_test = A_[val_idx]
        X_test = X[val_idx]
        Y_test = to_categorical(Y)[val_idx]

        # inputs = Input(shape=(G.shape[1:]))
        A_in = Input(A_.shape[1:], name='A_in')
        X_in = Input(X.shape[1:], name='X_in')

        # x1 = Dropout(0.5)(inputs)
        x1 = MyGCN(64, activation='relu')([A_in, X_in])
        # x1 = Dropout(0.5)(x1)
        x1 = MyGCN(10, activation='sigmoid')([A_in, x1])
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

        tb_callback = keras.callbacks.TensorBoard(log_dir='./Graph/' + str(j), histogram_freq=0, write_grads=False,
                                                  write_graph=True, write_images=False)
        # reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
        #                                                        factor=0.1,
        #                                                        patience=3,
        #                                                        verbose=1)

        history = model.fit([A_train, X_train],
                            Y_train,
                            # G_train.shape[0],
                            epochs=200,
                            verbose=0,
                            validation_split=0.2,
                            callbacks=[tb_callback])
        preds = model.predict([A_test, X_test])
        acc = accuracy(preds, Y_test)

        print("\n\n\nacc: ", acc, "\n\n\n")

        accuracies.append(acc)

        # loss, metrics = model.evaluate(G_test, Y_test)

    print(dataset_name)
    print("mean acc: ", np.mean(accuracies))
    print("std dev:", np.std(accuracies))
    print("accs:", accuracies)
