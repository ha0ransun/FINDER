#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from FINDER import FINDER
from tqdm import tqdm



def main():
    dqn = FINDER()
    graph_types = ['barabasi_albert']
    cost_types = ['uniform']
    heur_types = ['HDA', 'HBA', 'HCA', 'HPRA']
    file_path = '../../results/FINDER_ND/synthetic'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    for graph_type in graph_types:
        for cost_type in cost_types:
            for heur in heur_types:
                data_test_path = '../../data/synthetic/{}/{}/'.format(graph_type, cost_type)
                data_test_name = ['30-50', '50-100', '100-200', '200-300', '300-400', '400-500']

                with open('%s/%s_%s_%s_score.txt' % (file_path, graph_type, heur, cost_type), 'w') as fout:
                    for i in tqdm(range(len(data_test_name))):
                        data_test = data_test_path + data_test_name[i]
                        score_mean, score_std, time_mean, time_std = dqn.EvaluateHeuristics(data_test, heur)
                        fout.write('%.2f±%.2f,' % (score_mean * 100, score_std * 100))
                        fout.write('%.2f±%.2f\n' % (time_mean, time_std))
                        fout.flush()
                        print(100 * '#')
                        print('data_test_{} has been tested!'.format(data_test_name[i]))


if __name__=="__main__":
    main()
