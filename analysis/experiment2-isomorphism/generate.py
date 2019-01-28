import networkx as nx
import numpy as np

def generate_graph_samples(initial_graph, number_of_graphs, seed=42):
    # Seed
    sampled_graphs = []
    graph = generate_one_graph_sample(initial_graph, seed)

    sampled_graphs.append(graph)

    for i in range(number_of_graphs - 1):
        sampled_graphs.append(permute_graph(graph, seed=i))

    return sampled_graphs


def generate_one_graph_sample(initial_graph, seed=42):
    degree_sequence = [d for n, d in initial_graph.degree()]
    sample_graph = nx.random_degree_sequence_graph(degree_sequence, seed=seed)

    return sample_graph


def permute_graph(graph, seed):
    list_nodes = list(graph.nodes())
    np.random.seed(seed)
    permuted_list_nodes = np.random.permutation(list_nodes)
    new_mapping = {i: j for i, j in zip(list_nodes, permuted_list_nodes)}
    H = nx.relabel_nodes(graph, new_mapping)

    return H


def generate_samples_of_n_classes(num_nodes, num_class, num_graphs_per_class, p=0.15, seed=42):
    initial_graph = nx.generators.erdos_renyi_graph(num_nodes, p, seed)
    num_discarded_graphs = 0
    while not nx.is_connected(init_graph):
        initial_graph = nx.generators.erdos_renyi_graph(num_nodes, p)
        # print(nx.number_connected_components(init_graph))
        num_discarded_graphs += 1
    print('number of discarded graphs: {}'.format(num_discarded_graphs))
    assert (nx.is_connected(init_graph)), 'graph is not fully connected'

    dict_class_graphs = {}
    for i in range(num_class):
        list_graphs = generate_graph_samples(initial_graph, num_graphs_per_class, seed=i)
        dict_class_graphs[i] = list_graphs

    return dict_class_graphs


graph_dataset = generate_samples_of_n_classes(num_nodes=num_nodes, num_class=num_classes,
                                              num_graphs_per_class=num_graphs_per_class)