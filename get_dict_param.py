import sys
# add one more path ? try to
sys.path.append("..")
import numpy as np
import scipy as sp
import scipy.sparse as sps
import matplotlib.pyplot as plt
import random
from scipy import stats
from scipy.optimize import fmin
import os
import zipfile
from functools import partial
import time
from skopt.space import Real, Integer, Categorical
from Notebooks_utils.data_splitter import train_test_holdout
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Notebooks_utils.evaluation_function import evaluate_algorithm
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_ElasticNet
from GraphBased.P3alphaRecommender import P3alphaRecommender


def rowSplit(rowString, numColumns):
    split = rowString.split(",")
    split[numColumns-1] = split[numColumns-1].replace("\n", "")
    # change the type of data
    for i in range(numColumns-1):
        split[i] = int(split[i])
    split[numColumns-1] = float(split[numColumns-1])
    result = tuple(split)
    return result

def get_URM(URM_path):
    with open(URM_path, 'r') as URM_file:
        URM_file.seek(1)
        URM_tuples = []
        index = 0
        for line in URM_file:
            if index:
                URM_tuples.append(rowSplit(line, 3))
            index = index + 1
        print("Print out the first tripple: ")
        print(URM_tuples[0])

        userList, itemList, ratingList = zip(*URM_tuples)

        userList = list(userList)
        itemList = list(itemList)
        ratingList = list(ratingList)

        URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
        URM_all = URM_all.tocsr()
    return URM_all


def rowSplit_ICM(rowString):
    split = rowString.split(",")
    split[2] = split[2].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = float(split[2])  # tag is a string, not a float like the rating

    result = tuple(split)

    return result


def get_ICM(ICM_path, URM_all):
    ICM_sub_class_file.seek(0)
    ICM_sub_class_tuples = []

    index_line = 0
    for line in ICM_sub_class_file:
        if index_line:
            ICM_sub_class_tuples.append(rowSplit_ICM(line))
        index_line = index_line + 1

    itemList_icm_sub_class, featureList_icm_sub_class, dataList_icm_sub_class = zip(*ICM_sub_class_tuples)


    n_items = URM_all.shape[1]
    n_features = max(featureList_icm_sub_class) + 1

    ICM_shape = (n_items, n_features)

    ones = np.ones(len(dataList_icm_sub_class))
    ICM_all = sps.coo_matrix((ones, (itemList_icm_sub_class, featureList_icm_sub_class)), shape=ICM_shape)
    ICM_all = ICM_all.tocsr()
    return ICM_all, n_items, n_features

data_file_path = "./data"
URM_path = data_file_path + "/data_train.csv"
test_path = data_file_path + "/data_target_users_test.csv"
URM_all = get_URM(URM_path)


ICM_asset_path = "data/data_ICM_asset.csv"
ICM_asset_file = open(ICM_asset_path, 'r')

ICM_price_path = "data/data_ICM_price.csv"
ICM_price_file = open(ICM_price_path, 'r')

ICM_sub_class = "data/data_ICM_sub_class.csv"
ICM_sub_class_file = open(ICM_sub_class, 'r')


ICM_all, n_items, n_features = get_ICM(ICM_sub_class, URM_all)
print("Number of items is ", str(n_items))
print("n_features is ", str(n_features))


# The P3alpha parameter
topKList_P3 = [20, 64, 29]
alphaList = [0.5522912790456566, 0.5626527178823623,
             0.361456057829241, ]
minRateList = [0.03083106478857079, 0.4999280105627021,
               0.0035879242058026737]
implicitList = [False, False, False]

#(topK=topK_P3, alpha=alphaList[i], min_rating=minRateList[i], implicit=implicitList[i])
p3_param_name = ['topK', 'alpha', 'min_rating', 'implicit']
p3_params_list = []
for i in range(len(topKList_P3)):
    p3_param = dict()
    p3_param[p3_param_name[0]] = topKList_P3[i]
    p3_param[p3_param_name[1]] = alphaList[i]
    p3_param[p3_param_name[2]] = minRateList[i]
    p3_param[p3_param_name[3]] = implicitList
    p3_params_list.append(p3_param)

print(p3_params_list)


# Store the value
loopTimes=10
p3_MAPs = np.zeros([len(p3_params_list), loopTimes + 1])

for j in range(loopTimes):
    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.8)
    for i, p3_param in enumerate(p3_params_list):
        if p3_MAPs[i][loopTimes] == 1:
            continue
        recommender = P3alphaRecommender(URM_train)
        recommender.fit(**p3_param)
        evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10])
        eval_res = evaluator_validation.evaluateRecommender(recommender)
        MAP = eval_res[0][10]['MAP']
        # MAP = 1
        print("The MAP for {} th params is {}".format(i, MAP))
        p3_MAPs[i][j] = MAP
        if MAP < 0.02:
            p3_MAPs[i][10] = 1

print(p3_MAPs)
np.savetxt("./evalRes/P3alpha.csv", p3_MAPs, delimiter=",")


# The Slim parameters
topKList_SLIM = [23, 21, 20, 22, 20, 20]
epochList = [999, 999, 499, 999, 199, 199]
lambdaIList = [0.0000121296054612313, 0.005255993,
               0.00001, 0.008363589,
               0.00001, 0.00001]
lambdaJList = [0.0000100325008116582, 0.0000977541952150208,
               0.01, 0.003058426, 0.01, 0.00001]
lrList = [0.001261884, 0.000112325, 0.001472739,
          0.000102219, 0.001275566, 0.0001]
bsList = [2000, 1000, 500, 1500, 500, 500]

slim_params_list = []
slim_my_param1 = {'topK': 9, 'epochs': 249, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 4.9166438185561244e-05, 'lambda_j': 0.006660043170232182, 'learning_rate': 0.0001025620204810499}
slim_my_param2 = {'topK': 5, 'epochs': 199, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 0.01, 'lambda_j': 0.01, 'learning_rate': 0.0001}
slim_my_param3 = {'topK': 1000, 'epochs': 199, 'symmetric': True, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001}
slim_my_param4 = {'topK': 1000, 'epochs': 199, 'symmetric': True, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 1e-05, 'learning_rate': 0.0001}

slim_params_list.append(slim_my_param1)
slim_params_list.append(slim_my_param2)
slim_params_list.append(slim_my_param3)
slim_params_list.append(slim_my_param4)


slim_param_name = ['topK', 'epochs', 'sgd_mode', 'symmetric', 'lambda_i', 'lambda_j', 'learning_rate', 'batch_size']
for i in range(len(topKList_SLIM)):
    slim_param = dict()
    slim_param[slim_param_name[0]] = topKList_SLIM[i]
    slim_param[slim_param_name[1]] = epochList[i]
    slim_param[slim_param_name[2]] = 'adagrad'
    slim_param[slim_param_name[3]] = False
    slim_param[slim_param_name[4]] = lambdaIList[i]
    slim_param[slim_param_name[5]] = lambdaJList[i]
    slim_param[slim_param_name[6]] = lrList[i]
    slim_param[slim_param_name[7]] = bsList[i]
    slim_params_list.append(slim_param)


#for i in slim_params_list:
#    print(i)

# Store the value
loopTimes=10
slim_MAPs = np.zeros([len(slim_params_list), loopTimes + 1])

for j in range(loopTimes):
    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.8)
    for i, slim_param in enumerate(slim_params_list):
        if slim_MAPs[i][loopTimes] == 1:
            continue
        recommender = SLIM_BPR_Cython(URM_train)
        recommender.fit(**slim_param)
        evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10])
        eval_res = evaluator_validation.evaluateRecommender(recommender)
        MAP = eval_res[0][10]['MAP']
        # MAP = 1
        print("The MAP for {} th params is {}".format(i, MAP))
        slim_MAPs[i][j] = MAP
        if MAP < 0.02:
            slim_MAPs[i][10] = 1

print(slim_MAPs)
np.savetxt("./evalRes/slim.csv", slim_MAPs, delimiter=",")


# Define the combine ratio
lineParam = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]