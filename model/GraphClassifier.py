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
from Models.GCN.gcn_utils import *
import keras


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


# prepare data
print("preparing data")
mutag = dl.DropboxLoader("NCI1")
A, X = get_A_X(mutag)
G = get_G(A, X)
Y = mutag.get_graph_label()

Y['graph_label'] = Y['graph_label'].apply(lambda x: 1 if x == 1 else 0)
Y = np.array(Y)

folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=2).split(G, Y))

accuracies = []
for j, (train_idx, val_idx) in enumerate(folds):
    print("split :", j)
    G_train = G[train_idx]
    Y_train = to_categorical(Y)[train_idx]
    G_test = G[val_idx]
    Y_test = to_categorical(Y)[val_idx]

    inputs = Input(shape=(G.shape[1:]))

    # x1 = Dropout(0.5)(inputs)
    x1 = MyGCN(64, activation='relu')(inputs)
    # x1 = Dropout(0.5)(x1)
    x1 = MyGCN(1, activation='sigmoid')(x1)
    x1 = Lambda(lambda G: G[:, :, G.shape[1]:])(x1)
    x1 = Permute((2, 1))(x1)
    # x1 = Dropout(0.5)(x1)

    # x2 = Dropout(0.5)(inputs)
    x2 = MyGCN(64, activation='relu')(inputs)
    # x2 = Dropout(0.5)(x2)
    x2 = MyGCN(2, activation='relu')(x2)
    x2 = Lambda(lambda G: G[:, :, G.shape[1]:])(x2)
    # x2 = Dropout(0.5)(x2)

    x3 = Dot(axes=(2, 1))([x1, x2])
    x3 = Activation(activation='softmax', input_shape=(1, 2))(x3)
    x3 = Reshape((2,))(x3)

    model = Model(inputs=inputs, outputs=x3)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy')
    # model.summary()

    history = model.fit(G_train, Y_train, G_train.shape[0], epochs=200, verbose=1)
    preds = model.predict(G_test)
    acc = accuracy(preds, Y_test)

    print("\n\n\nacc: ", acc, "\n\n\n")

    accuracies.append(acc)

    # loss, metrics = model.evaluate(G_test, Y_test)

print("accs:", accuracies)
print("mean acc: ", np.mean(accuracies))
