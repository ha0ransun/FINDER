#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from FINDER import FINDER
import numpy as np
from tqdm import tqdm
import time
import networkx as nx
import pandas as pd
import pickle as cp
import random


def GetSolution(STEPRATIO, MODEL_FILE):
    ######################################################################################################################
    ##................................................Get Solution (model).....................................................
    dqn = FINDER()
    ## data_test
    data_test_path = '../../data/real/'
#     data_test_name = ['Crime', 'HI-II-14', 'Digg', 'Enron', 'Gnutella31', 'Epinions', 'Facebook', 'Youtube', 'Flickr']
    data_test_name = ['Crime', 'HI-II-14', 'Facebook']
    data_test_costType = ['degree', 'random']
    model_file_path = './models/Model_barabasi_albert/'
    model_file_ckpt = MODEL_FILE
    model_file = model_file_path + model_file_ckpt
    ## save_dir
    save_dir = '../../results/FINDER_CN_cost/real/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    save_dir_degree = save_dir + '/Data_degree'
    save_dir_random = save_dir + '/Data_random'
    os.mkdir(save_dir_degree)
    os.mkdir(save_dir_random)

    ## begin computing...
    print('The best model is :%s' % (model_file))
    dqn.LoadModel(model_file)

    for costType in data_test_costType:
        df = pd.DataFrame(np.arange(2 * len(data_test_name)).reshape((-1, len(data_test_name))), index=['time', 'score'],
                          columns=data_test_name)
        #################################### modify to choose which stepRatio to get the solution
        stepRatio = STEPRATIO
        for j in range(len(data_test_name)):
            print('Testing dataset %s' % data_test_name[j])
            data_test = data_test_path + data_test_name[j] + '_' + costType + '.gml'
            if costType == 'degree':
                solution, time, robustness = dqn.EvaluateRealData(model_file, data_test, save_dir_degree, stepRatio)
            elif costType == 'random':
                solution, time, robustness = dqn.EvaluateRealData(model_file, data_test, save_dir_random, stepRatio)
            df.iloc[0, j] = time
            df.iloc[1, j] = robustness
            print('Data:%s, cost:%s, time:%.2f, score:%.2f'%(data_test_name[j], costType, time, robustness))
        if costType == 'degree':
            save_dir_local = save_dir_degree + '/StepRatio_%.4f' % stepRatio
        elif costType == 'random':
            save_dir_local = save_dir_random + '/StepRatio_%.4f' % stepRatio
        if not os.path.exists(save_dir_local):
            os.mkdir(save_dir_local)

        df.to_csv(save_dir_local + '/solution_%s_time.csv' % costType, encoding='utf-8', index=False)
        print('model has been tested!')


def EvaluateSolution(STEPRATIO, STRTEGYID):
    #######################################################################################################################
    ##................................................Evaluate
    dqn = FINDER()
    ## data_test
    data_test_path = '../data/real/Cost/'
#     data_test_name = ['Crime', 'HI-II-14', 'Digg', 'Enron', 'Gnutella31', 'Epinions', 'Facebook', 'Youtube', 'Flickr']
    data_test_name = ['Crime', 'HI-II-14']
    data_test_costType = ['degree', 'random']

    ## save_dir
    save_dir_degree = '../results/FINDER_CN_cost/real/Data_degree/StepRatio_%.4f/' % STEPRATIO
    save_dir_random = '../results/FINDER_CN_cost/real/Data_random/StepRatio_%.4f/' % STEPRATIO
    ## begin computing...

    for costType in data_test_costType:
        df = pd.DataFrame(np.arange(2 * len(data_test_name)).reshape((2, len(data_test_name))),
                          index=['solution', 'time'], columns=data_test_name)
        for i in range(len(data_test_name)):
            print('Evaluating dataset %s' % data_test_name[i])
            data_test = data_test_path + data_test_name[i] + '_' + costType + '.gml'
            if costType == 'degree':
                solution = save_dir_degree + data_test_name[i] + '_degree.txt'
            elif costType == 'random':
                solution = save_dir_random + data_test_name[i] + '_random.txt'

            t1 = time.time()
            # strategyID: 0:no insert; 1:count; 2:rank; 3:multiply
            ################################## modify to choose which strategy to evaluate
            strategyID = STRTEGYID
            score, MaxCCList, solution = dqn.EvaluateSol(data_test, solution, strategyID, reInsertStep=20)
            t2 = time.time()
            df.iloc[0, i] = score
            df.iloc[1, i] = t2 - t1
            if costType == 'degree':
                result_file = save_dir_degree + '/MaxCCList__Strategy_' + data_test_name[i] + '.txt'
            elif costType == 'random':
                result_file = save_dir_random + '/MaxCCList_Strategy_' + data_test_name[i] + '.txt'

            with open(result_file, 'w') as f_out:
                for j in range(len(MaxCCList)):
                    f_out.write('%.8f\n' % MaxCCList[j])

            print('Data:%s, score:%.6f!' % (data_test_name[i], score))

        if costType == 'degree':
            df.to_csv(save_dir_degree + '/solution_%s_score.csv' % (costType), encoding='utf-8', index=False)
        elif costType == 'random':
            df.to_csv(save_dir_random + '/solution_%s_score.csv' % (costType), encoding='utf-8', index=False)


def main():
    model_file = 'nrange_30_50_iter_399000_barabasi_albert.ckpt'
    GetSolution(0.01, model_file)
    # EvaluateSolution(0.01, 0)

if __name__=="__main__":
    main()