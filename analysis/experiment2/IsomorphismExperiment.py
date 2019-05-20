import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from model.GraphClassifier import GraphClassifier
from analysis.experiment2 import generate
from model.GCNWithOptionalSum import GCNWithOptionalSum

from model.WLKernel import WLKernel

from csv import writer


class IsomorphismExperiment(object):

    def name(self):
        return 'GC'

    def get_accuracies(self, A, X, Y, num_graph_classes, splits=None, batch_size=None):
        classifier = GraphClassifier()

        accs, times = classifier.fit_eval(A, X, Y, num_classes=num_graph_classes,
                                          epochs=200, batch_size=batch_size, folds=splits,
                                          preprocess_A=None, verbose=1)
        return accs

    def main(self):

        out = open('out.csv', 'a')
        res_writer = writer(out, delimiter=";")
        res_writer.writerow(['classifier', 'exs_per_class', 'mean_acc', 'std_acc', 'accs'])

        num_nodes_per_graph = 50
        num_graph_classes = 5
        num_node_classes = 2
        num_graphs_per_class = 100
        batch_size = 5
        examples_per_classes = [18, 20]

        classifiers = [WLKernel(), GCNWithOptionalSum(True), GCNWithOptionalSum(False), self]

        # generate data
        A, X, Y = generate.get_tensors(num_nodes_per_graph,
                                       num_graph_classes,
                                       num_node_classes,
                                       num_graphs_per_class)

        for classifier in classifiers:
            print("classifier:", classifier)

            for exs_per_class in examples_per_classes:
                splits = StratifiedShuffleSplit(n_splits=10, train_size=num_graph_classes * exs_per_class).split(X, Y)

                accs = classifier.get_accuracies(A, X, Y, num_graph_classes, splits=splits, batch_size=batch_size)
                print("acc:", np.mean(accs), accs)
                res_writer.writerow([classifier.name(), exs_per_class, np.mean(accs), np.std(accs), accs])
                out.flush()


if __name__ == '__main__':
    IsomorphismExperiment().main()
