import os
import pickle
from math import ceil

import pandas as pd
import numpy as np
import networkx as nx
from ImportData import DropboxLoader
from scipy.sparse import csr_matrix


def get_A_X(loader, normalise_by_num_nodes=True):
    all_adj = loader.get_adj()
    graph_ind = loader.get_graph_ind()
    numGraphs = graph_ind.max()['graph_ind']

    def get_A_X_by_graph_id(id):
        nodes_to_keep = graph_ind.loc[graph_ind['graph_ind'] == id]['node']
        adj = all_adj[all_adj['from'].isin(nodes_to_keep) | all_adj['to'].isin(nodes_to_keep)]

        all_X = loader.get_node_label()
        ids = all_X[all_X['node'].isin(nodes_to_keep)].index
        X = pd.get_dummies(all_X['label']).iloc[ids].values
        # G = nx.from_pandas_edgelist(adj, 'from', 'to')
        g = nx.Graph()
        g.add_nodes_from(ids)
        g.add_edges_from(adj.values)
        A = nx.to_numpy_array(g)
        return A, X

    A, X = zip(*[get_A_X_by_graph_id(id) for id in range(1, numGraphs + 1)])
    num_nodes_list = list(map(lambda a: a.shape[0], X))
    maxNodes = max(num_nodes_list)

    def normaliseX_by_num_nodes(x):
        return np.divide(x, x.shape[0])

    def padA(a):
        padA = np.zeros([maxNodes, maxNodes])
        padA[:a.shape[0], :a.shape[1]] = a
        return padA

    def padX(x):
        padX = np.zeros([maxNodes, x.shape[1]])
        padX[:x.shape[0], :x.shape[1]] = x
        return padX

    padA = list(map(padA, A))
    if normalise_by_num_nodes:
        X = map(normaliseX_by_num_nodes, X)

    padX = list(map(padX, X))

    return padA, padX


def get_data(dropbox_name, normalise_by_num_nodes=False):
    dataset = DropboxLoader(dropbox_name)

    file_path = f'{dropbox_name}.p'

    if os.path.isfile(file_path):
        A_list, X = pickle.load(open(file_path, "rb"))
    else:
        A, X = get_A_X(dataset, normalise_by_num_nodes)

        A_list = list(map(csr_matrix, A))
        pickle.dump((A_list, X), open(file_path, "wb"))
    Y = dataset.get_graph_label()

    return A_list, X, Y


def batch_generator(self, A_X, y, batch_size):
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


def pred_batch_generator(self, A_X, y, batch_size):
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
