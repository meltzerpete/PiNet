import networkx as nx
import numpy as np

NODE_LABELS = 'feats'


def generate_graph_samples(initial_graph, number_of_graphs, num_node_classes, seed=42):
    # Seed
    sampled_graphs = []
    graph = generate_one_graph_sample(initial_graph, seed)

    # generate random node features
    node_classes = np.random.random_integers(0, num_node_classes - 1, nx.number_of_nodes(initial_graph))

    feats = {k: v for k, v in enumerate(node_classes)}
    nx.set_node_attributes(graph, feats, NODE_LABELS)

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


def generate_samples_of_n_classes(num_nodes, num_class, num_graphs_per_class, num_node_classes, p=0.15, seed=42):
    initial_graph = nx.generators.erdos_renyi_graph(num_nodes, p, seed)

    num_discarded_graphs = 0
    while not nx.is_connected(initial_graph):
        initial_graph = nx.generators.erdos_renyi_graph(num_nodes, p)
        # print(nx.number_connected_components(init_graph))
        num_discarded_graphs += 1
    print('number of discarded graphs: {}'.format(num_discarded_graphs))
    assert (nx.is_connected(initial_graph)), 'graph is not fully connected'

    dict_class_graphs = {}
    for i in range(num_class):
        list_graphs = generate_graph_samples(initial_graph, num_graphs_per_class, num_node_classes, seed=i)
        dict_class_graphs[i] = list_graphs

    return dict_class_graphs


def gen_iter_attributes_dict(nx_graph, num_features, seed=42, attr_name='node_label'):
    #    for i in df_nodes.label.unique():
    #        yield(i,df_nodes[df_nodes['label']==i].node.tolist())

    num_nodes = nx_graph.number_of_nodes()
    nodes = list(nx_graph.nodes())
    random_features = np.random.choice(range(num_features), num_nodes)

    for node, value in zip(nodes, random_features):
        yield (node, {attr_name: value})


def add_random_categorical_variables(nx_graph, num_features):
    nx_graph.add_nodes_from(gen_iter_attributes_dict(nx_graph, num_features))







def get_tensors(num_nodes_per_graph, num_graph_classes, num_node_classes, num_graphs_per_class):
    graph_dataset = generate_samples_of_n_classes(num_nodes=num_nodes_per_graph,
                                                  num_class=num_graph_classes,
                                                  num_node_classes=num_node_classes,
                                                  num_graphs_per_class=num_graphs_per_class)

    A = []
    X = []
    Y = []

    for n in range(num_graph_classes):
        for g_id in range(num_graphs_per_class):
            A.append(nx.to_numpy_array(graph_dataset[n][g_id], range(num_nodes_per_graph)))
            x_kv = nx.get_node_attributes(graph_dataset[n][g_id], NODE_LABELS )
            x = np.zeros([num_nodes_per_graph, num_node_classes])
            for r, c in x_kv.items():
                x[r, c] = 1
            X.append(x)

            Y.append(n)
    Y = np.array(Y).reshape([len(Y), 1])

    return np.array(A), np.array(X), Y

# A, X, Y = get_tensors(num_nodes_per_graph=5, num_graph_classes=3, num_node_classes=5, num_graphs_per_class=3)
