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
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Notebooks_utils.data_splitter import train_test_holdout
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython


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



URM_train, URM_test = train_test_holdout(URM_all, train_perc = 0.8)
URM_train, URM_validation = train_test_holdout(URM_train, train_perc = 0.9)




evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


recommender_class = MatrixFactorization_FunkSVD_Cython


output_folder_path = "result_experiments/"


# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)


evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

#runParameterSearch_Collaborative(UserKNNCFRecommender, URM_train, URM_test, metric_to_optimize="MAP",
#                                 n_cases=100, evaluator_validation=evaluator_validation,
#                                 evaluator_test=evaluator_test, )

output_folder_path = "result_experiments/"


# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

metric_to_optimize = "MAP"

runParameterSearch_Collaborative(recommender_class, URM_train=URM_train,
                                 metric_to_optimize=metric_to_optimize, evaluator_validation=evaluator_validation,
                                 evaluator_test=evaluator_test)


data_loader = DataIO(folder_path = output_folder_path)
search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")

print(search_metadata)

parameterSearchResultPath = output_folder_path + "ParameterSearchResult/"
if not os.path.exists(parameterSearchResultPath):
    os.makedirs(parameterSearchResultPath)

with open(parameterSearchResultPath+recommender_class.RECOMMENDER_NAME+'search_metadata.txt', 'w') as f:
    f.write(json.dumps(search_metadata))

best_parameters = search_metadata["hyperparameters_best"]
print(best_parameters)

with open(parameterSearchResultPath+recommender_class.RECOMMENDER_NAME+'hyperparameters_best.txt', 'w') as f:
    f.write(json.dumps(best_parameters))