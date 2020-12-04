import networkx as nx
import numpy as np
import random



data_path = '../../data/real/'
data_name = ['Crime','HI-II-14', 'Facebook']
# save_dir = './Data/InternetP2P_cost/'
costType = {0: 'degree', 1:'random'}

for k in range(2):
    for i in range(len(data_name)):
        data = data_path + data_name[i] + '.txt'
        g = nx.read_weighted_edgelist(data)

        if k == 0:  ### degree weight
            g_add = g.copy()
            degree = dict(nx.degree(g))
            maxDegree = max(degree.values())
            weights = {}
            for node in g.nodes():
                weights[node] = degree[node] / maxDegree

        else:  ### random weight
            g_add = g.copy()
            weights = {}
            for node in g.nodes():
                weights[node] = random.uniform(0, 1)

        nx.set_node_attributes(g_add, weights, 'weight')
        save_dir_g = '%s/%s_%s.gml' % (data_path, data_name[i], costType[k])
        nx.write_gml(g_add, save_dir_g)