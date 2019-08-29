#!/opt/conda/bin/python
import os
import pickle
import sys
from csv import writer

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from ImportData import DropboxLoader
from model.GCNWithOptionalMean import GCNWithOptionalMean
from model.PiNet import PiNet
from model.WLKernel import WLKernel


class Benchmark(object):

    def name(self):
        return 'PiNet'

    def get_accs_times(self, A, X, Y, num_graph_classes, splits=None, batch_size=None):
        classifier = PiNet()

        accs, times = classifier.fit_eval(A, X, Y, num_classes=num_graph_classes,
                                          epochs=200, batch_size=batch_size, folds=splits,
                                          verbose=0)
        return accs, times

    def main(self):

        datasets = {
            'MUTAG': {},
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
            'PROTEINS': {},
            'NCI1': {
                'pretty_name': 'NCI-1',
            },
            'NCI109': {
                'pretty_name': 'NCI-109',
            },
        }

        if len(sys.argv) > 1:
            args = sys.argv[1:]
            datasets = {
                k: v for k, v in map(lambda arg: (arg, datasets[arg]), args)
            }

        for dataset_name, dataset in datasets.items():

            print("Dataset: ", dataset_name)

            dropbox_name = dataset['dropbox_name'] \
                if 'dropbox_name' in dataset.keys() else dataset_name

            pretty_name = dataset['pretty_name'] \
                if 'pretty_name' in dataset.keys() else dataset_name

            classes = dataset['classes'] \
                if 'classes' in dataset.keys() else 2

            out = open('out.csv', 'a')
            res_writer = writer(out, delimiter=";")
            res_writer.writerow(
                ["classifier", "dataset", "pretty_name", "mean_acc", "acc_std",
                 "mean_train_time(s)", "time_std", "all_accs", "all_times"])

            classifiers = [self, WLKernel(), GCNWithOptionalMean(with_mean=True), GCNWithOptionalMean(with_mean=False)]

            # generate data
            A, X, Y = get_data(dropbox_name)

            Y['graph_label'] = Y['graph_label'].apply(lambda x: 1 if x == 1 else 0, 0)

            for classifier in classifiers:
                print("classifier:", classifier.name(), flush=True)

                splits = StratifiedShuffleSplit(n_splits=10).split(X, Y)

                accs, times = classifier.get_accs_times(A, X, Y, classes, splits=splits, batch_size=50)
                print("acc:", np.mean(accs), accs)
                res_writer.writerow(
                    [classifier.name(), dataset, pretty_name, np.mean(accs), np.std(accs),
                     np.mean(times), np.std(times), accs, times])
                out.flush()


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


if __name__ == '__main__':
    Benchmark().main()
