from urllib.request import urlretrieve
import zipfile, os
import time
import numpy as np
import os
from Notebooks_utils.data_splitter import train_test_holdout
from datetime import datetime
from Base.Evaluation.Evaluator import EvaluatorHoldout
import scipy.sparse as sps
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Notebooks_utils.data_splitter import train_test_holdout
from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
import json
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Base.DataIO import DataIO
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Real, Integer, Categorical
from GraphBased.RP3betaRecommender import RP3betaRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from KNN.UserKNNCFRecommender import UserKNNCFRecommender


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


from Notebooks_utils.data_splitter import train_test_holdout

URM_train, URM_test = train_test_holdout(URM_all, train_perc = 0.8)
URM_train, URM_validation = train_test_holdout(URM_train, train_perc = 0.9)

sps.save_npz("./result_experiments/URM_all.npz", URM_all)
sps.save_npz("./result_experiments/URM_train.npz", URM_train)
sps.save_npz("./result_experiments/URM_test.npz", URM_test)

slim_best_parameters = {'topK': 1000, 'epochs': 199, 'symmetric': True, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001}
rp3_best_parameters = {"topK": 44, "alpha": 0.0410753731972553, "beta": 0.019312081484212932, "normalize_similarity": True}
userKNNCF_best_parameters = {"topK": 466, "shrink": 9, "similarity": "dice", "normalize": False}


URM_train = sps.csr_matrix(URM_train)
profile_length = np.ediff1d(URM_train.indptr)
block_size = int(len(profile_length)*0.05)
sorted_users = np.argsort(profile_length)

slim_model = SLIM_BPR_Cython(URM_train,recompile_cython=False)
slim_model.fit(**slim_best_parameters)
rp3_model = RP3betaRecommender(URM_train)
rp3_model.fit(**rp3_best_parameters)
userCF_model = UserKNNCFRecommender(URM_train)
userCF_model.fit(**userKNNCF_best_parameters)

MAP_slim_per_group = []
MAP_rp3_per_group = []
MAP_userCF_per_group = []
cutoff = 10


URM_train = sps.csr_matrix(URM_train)
profile_length = np.ediff1d(URM_train.indptr)
block_size = int(len(profile_length)*0.05)
sorted_users = np.argsort(profile_length)

for group_id in range(0, 20):
    start_pos = group_id * block_size
    end_pos = min((group_id + 1) * block_size, len(profile_length))

    users_in_group = sorted_users[start_pos:end_pos]

    users_in_group_p_len = profile_length[users_in_group]

    print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                  users_in_group_p_len.mean(),
                                                                  users_in_group_p_len.min(),
                                                                  users_in_group_p_len.max()))

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=users_not_in_group)

    results, _ = evaluator_test.evaluateRecommender(slim_model)
    MAP_slim_per_group.append(results[cutoff]["MAP"])

    results, _ = evaluator_test.evaluateRecommender(rp3_model)
    MAP_rp3_per_group.append(results[cutoff]["MAP"])

    results, _ = evaluator_test.evaluateRecommender(userCF_model)
    MAP_userCF_per_group.append(results[cutoff]["MAP"])

slim_model.save_model("./result_experiments/results_ensemble/", "slim_1")
rp3_model.save_model("./result_experiments/results_ensemble/", "rp3_1")
userCF_model.save_model("./result_experiments/results_ensemble/", "userCF_1")


import matplotlib.pyplot as pyplot

pyplot.plot(MAP_slim_per_group, label="slim")
pyplot.plot(MAP_rp3_per_group, label="rp3")
pyplot.plot(MAP_userCF_per_group, label="userCF")
pyplot.ylabel('MAP')
pyplot.xlabel('User Group')
pyplot.legend()
pyplot.show()
