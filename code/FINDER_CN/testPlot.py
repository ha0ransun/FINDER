#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys,os
import networkx as nx
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from FINDER import FINDER
from tqdm import tqdm



def main():
    dqn = FINDER()
    graph_types = ['barabasi_albert']
    cost_types = ['uniform']
    heur_types = ['RL', 'HDA', 'HBA', 'HCA', 'HPRA']
    file_path = '../../plots/FINDER_CN/synthetic'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    data_test_name = '50-100' # ['30-50', '50-100', '100-200', '200-300', '300-400', '400-500']

    for i in range(100):
        for graph_type in graph_types:
            for cost_type in cost_types:
                data_test_path = '../../data/synthetic/{}/{}/'.format(graph_type, cost_type)
                model_file = f'./models/Model_{graph_type}/nrange_30_50_iter_399000_{graph_type}.ckpt'
                data_test = data_test_path + data_test_name
                g_path = '%s/'%data_test + f'g_{i}' # could be changed for other g_i
                g = nx.read_gml(g_path)
                for heur in heur_types:
                    dqn.Evaluate1(g, i, file_path, model_file, heur)


if __name__=="__main__":
    main()
