#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from FINDER import FINDER
from tqdm import tqdm



def main():
    dqn = FINDER()
    graph_types = ['barabasi_albert', 'erdos_renyi', 'small-world']
    cost_types = ['uniform']
    file_path = '../../results/FINDER_CN/synthetic'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    for graph_type in graph_types:
        for cost_type in cost_types:
            data_test_path = '../../data/synthetic/{}/{}/'.format(graph_type, cost_type)
            data_test_name = ['30-50', '50-100', '100-200', '200-300', '300-400', '400-500']
            model_file = f'./models/Model_{graph_type}/nrange_30_50_iter_37500_{graph_type}.ckpt'

            with open('%s/%s_%s_score.txt' % (file_path, graph_type, cost_type), 'w') as fout:
                for i in tqdm(range(len(data_test_name))):
                    data_test = data_test_path + data_test_name[i]
                    score_mean, score_std, time_mean, time_std = dqn.Evaluate(data_test, model_file)
                    fout.write('%.2f±%.2f,' % (score_mean * 100, score_std * 100))
                    fout.flush()
                    print(100 * '#')
                    print('data_test_{} has been tested!'.format(data_test_name[i]))


if __name__=="__main__":
    main()
