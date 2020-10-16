import networkx as nx
import os
from numpy import random


def gen_graph(num_min, num_max, g_type, path, i, n_type='uniform'):
    cur_n = random.randint(num_max - num_min + 1) + num_min
    if g_type == 'erdos_renyi':
        g = nx.erdos_renyi_graph(n=cur_n, p=0.15)
    elif g_type == 'powerlaw':
        g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
    elif g_type == 'small-world':
        g = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
    elif g_type == 'barabasi_albert':
        g = nx.barabasi_albert_graph(n=cur_n, m=4)

    if n_type != 'uniform':
        if n_type == 'random':
            weights = {}
            for node in g.nodes():
                weights[node] = random.uniform(0, 1)
        else:
            degree = nx.degree_centrality(g)
            maxDegree = max(dict(degree).values())
            weights = {}
            for node in g.nodes():
                weights[node] = degree[node] / maxDegree
        nx.set_node_attributes(g, weights, 'weight')

    path = '%s/%s-%s' % (path, str(num_min), str(num_max))
    if not os.path.exists(path):
        os.mkdir(path)

    nx.write_gml(g, '%s/g_%s' % (path, str(i)))


graph_types = ['erdos_renyi', 'small-world', 'barabasi_albert']
cost_types = ['uniform', 'degree', 'random']
# data_size = [(30, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, 500)]
data_sizes = [(30, 50), (50, 100), (100, 200)]

if not os.path.exists('../data'):
    os.mkdir('../data')
if not os.path.exists('../data/synthetic'):
    os.mkdir('../data/synthetic')

for graph_type in graph_types:
    for cost_type in cost_types:
        for data_size in data_sizes:
            file_path = '../data/synthetic/%s' % (graph_type)
            if not os.path.exists(file_path):
                os.mkdir(file_path)

            file_path_sub = '%s/%s' % (file_path, cost_type)
            if not os.path.exists(file_path_sub):
                os.mkdir(file_path_sub)
            for size_min, size_max in data_sizes:
                for i in range(100):
                    gen_graph(size_min, size_max, graph_type, file_path_sub, i, cost_type)
