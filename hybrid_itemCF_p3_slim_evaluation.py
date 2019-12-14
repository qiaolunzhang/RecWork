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

from sklearn.linear_model import ElasticNet
import time
from skopt.space import Real, Integer, Categorical
import json
from Base.DataIO import DataIO
from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from Notebooks_utils.data_splitter import train_test_holdout
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Notebooks_utils.evaluation_function import evaluate_algorithm
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_ElasticNet
from GraphBased.P3alphaRecommender import P3alphaRecommender
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from datetime import datetime
from Base.NonPersonalizedRecommender import TopPop



# Define the hybrid class


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


def test_save(recommender, top_recommender, data_file_path, warm_users):
    test_path = data_file_path
    test_num = open(test_path, 'r')
    test_tuples = []
    idx = 0
    for aline in test_num:
        if idx:
            test_tuples.append(aline.replace("\n", ""))
        idx = idx + 1

    result = {}
    label = 0
    for user_idx in test_tuples:
        # for user_id in range(n_users_to_test):
        user_idx = int(user_idx)
        if user_idx in warm_users:
            # change warm_users id to the id in the recommender
            recommend_id = int(np.where(warm_users == user_idx)[0])
            rate = recommender.recommend(recommend_id, cutoff=10)
        else:
            rate = top_recommender.recommend(user_idx, cutoff=10)
        if user_idx % 3000 == 0:
            label = label + 1
            print('---------------{}0----------------'.format(label))
        result[user_idx] = rate
    return result


def create_csv(results, result_name, results_dir='./res/', ):

    csv_fname = result_name + datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:
        # masterList, temp = [], []
        # writer = csv.writer(f,delimiter=',')
        # for key, value in results.items():
        #     temp = [key, value]
        #     masterList.append(temp)
        #     writer.writerows(masterList)
        f.write('user_id,item_list\n')

        for key, value in results.items():
            nor_v = ""
            for i in value:
                nor_v = nor_v + str(i) + " "
            f.write(str(key) + ',' + nor_v + '\n')


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


class SimilarityHybridRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNSimilarityHybridRecommender
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNSimilarityHybridRecommender"


    def __init__(self, URM_train, Similarity_1, Similarity_2, Similarity_3, sparse_weights=True):
        super(SimilarityHybridRecommender, self).__init__(URM_train)

        if Similarity_1.shape != Similarity_2.shape:
            raise ValueError("ItemKNNSimilarityHybridRecommender: similarities have different size, S1 is {}, S2 is {}".format(
                Similarity_1.shape, Similarity_2.shape
            ))

        if Similarity_2.shape != Similarity_3.shape:
            raise ValueError("ItemKNNSimilarityHybridRecommender: similarities have different size, S1 is {}, S2 is {}".format(
                Similarity_2.shape, Similarity_3.shape
            ))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(Similarity_1.copy(), 'csr')
        self.Similarity_2 = check_matrix(Similarity_2.copy(), 'csr')
        self.Similarity_3 = check_matrix(Similarity_3.copy(), 'csr')


    def fit(self, topK=100, alpha1 = 0.5, alpha2 = 0.3, alpha3 = 0.2):

        self.topK = topK
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

        self.W_sparse = self.Similarity_1*self.alpha1 + self.Similarity_2*alpha2 + self.Similarity_3*alpha3


class ScoresHybridRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridRecommender"

    def __init__(self, URM_train, Recommender_1, Recommender_2):
        super(ScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2

    def fit(self, alpha=0.5):
        self.alpha = alpha

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * (1 - self.alpha)

        return item_weights


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

warm_users_mask = np.ediff1d(URM_all.tocsr().indptr) > 0
warm_users = np.arange(URM_all.shape[0])[warm_users_mask]





#itemCFParam = {'topK': 525, 'shrink': 0, 'similarity': 'cosine', 'normalize': True}
# The following is not the best one
#itemCFParam = {'topK': 671, 'shrink': 183, 'similarity': 'jaccard', 'normalize': True}
# This is the best one for itemCFParam
itemCFParam = {'topK': 12, 'shrink': 25, 'similarity': 'tversky', 'normalize': False}
slimParam = {'topK': 22, 'epochs': 999, 'sgd_mode': 'adagrad', 'symmetric': False, 'lambda_i': 0.008363589, 'lambda_j': 0.003058426, 'learning_rate': 0.000102219, 'batch_size': 1500}
p3Param = {'topK': 64, 'alpha': 0.5626527178823623, 'min_rating': 0.4999280105627021, 'implicit': [False, False, False]}

alpha1 = 0.4
alpha2 = 0.5399999999999999
alpha3 = 0.06000000000000005


print("***************************Ensure the parameter is good**********************")

URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.8)
itemCF_recommender = ItemKNNCFRecommender(URM_train)
itemCF_recommender.fit(**itemCFParam)
slim_recommender = SLIM_BPR_Cython(URM_train, recompile_cython=False)
slim_recommender.fit(**slimParam)
p3_recommender = P3alphaRecommender(URM_train)
p3_recommender.fit(**p3Param)

recommender1 = SimilarityHybridRecommender(URM_train, itemCF_recommender.W_sparse,
                                           slim_recommender.W_sparse, p3_recommender.W_sparse)
recommender1.fit(topK=100, alpha1=alpha1, alpha2=alpha2, alpha3=alpha3)

evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10])
eval_res = evaluator_validation.evaluateRecommender(recommender1)
MAP = eval_res[0][10]['MAP']
print("The MAP in one test is: ", MAP)


itemCF_recommender = ItemKNNCFRecommender(URM_all)
itemCF_recommender.fit(**itemCFParam)
slim_recommender = SLIM_BPR_Cython(URM_all, recompile_cython=False)
slim_recommender.fit(**slimParam)
p3_recommender = P3alphaRecommender(URM_all)
p3_recommender.fit(**p3Param)
recommender1 = SimilarityHybridRecommender(URM_all, itemCF_recommender.W_sparse,
                                           slim_recommender.W_sparse, p3_recommender.W_sparse)
recommender1.fit(topK=100, alpha1=alpha1, alpha2=alpha2, alpha3=alpha3)
recommender1.save_model("model/", "hybrid_item_slim_basic")

class TopUserTagPop(object):
    "TOP POP recommender grouped by the user tags"

    def __init__(self, URM_train, user_feature_matrix):
        self.user_feature = user_feature_matrix
        self.devided_top = np.zeros([8, 10, 18495])
        #self.devided_top = self.devided_top.sum(axis=0)
        self.URM_train = URM_train.toarray()

    def fit(self):
        for id in range(30911):
            user_region = self.user_feature[id][0]
            user_age = self.user_feature[id][1]
            if user_age == 0 or user_region == 0:
                continue
            else:
                user_region = user_region - 1
                user_age = user_age - 1
                self.devided_top[int(user_region), int(user_age), :] = \
                    self.devided_top[int(user_region), int(user_age), :] + self.URM_train[int(id), :]

    def recommend(self, user_id, cutoff=10):
        user_region = self.user_feature[int(user_id)][0]
        user_age = self.user_feature[int(user_id)][1]
        if user_age == 0 or user_region == 0:
            result = [17955, 8638, 5113, 10227, 4657, 197, 8982, 10466, 3922, 4361]
        else:
            user_region = user_region - 1
            user_age = user_age - 1
            devided_id_top = self.devided_top[int(user_region), int(user_age), :]
            devided_id_top = devided_id_top.sum(axis=0)
            devided_id_top = devided_id_top[int(user_age), :]
            top = devided_id_top.argsort()[-1*int(cutoff):][::-1]
            result = top.tolist()
        return result

def getUserLabel(data_path):
    num_users = 30911
    # col 0 stands for region, row 0 stands for age
    user_feature_matrix = np.zeros([num_users, 2])
    region_path = data_path + 'data_UCM_region.csv'
    age_path = data_path + 'data_UCM_age.csv'

    region_file = open(region_path, 'r')
    region_file.seek(1)
    index = 0
    for line in region_file:
        if index:
            userID, Region, feature = rowSplit(line)
            user_feature_matrix[userID][0] = Region + 1
        index = index + 1

    age_file = open(age_path, 'r')
    age_file.seek(1)
    index = 0
    for line in age_file:
        if index:
            userID, Age, feature = rowSplit(line)
            user_feature_matrix[userID][1] = Age
        index = index + 1

    return user_feature_matrix

user_feature_matrix = getUserLabel(data_file_path)
# topPopRecommender = GlobalEffects(URM_all)
# topPopRecommender.fit()
topPopRecommender = TopUserTagPop(URM_all, user_feature_matrix)
topPopRecommender.fit()

#topPopRecommender = TopPop(URM_all)
#topPopRecommender.fit()


results_test = test_save(recommender1, topPopRecommender, test_path, warm_users)
create_csv(results_test, '20191214_hybrid_itemCF_p3_slim')
print("finished")

