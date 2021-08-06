#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import statistics
import math
from sklearn.model_selection import train_test_split
import random
import sklearn
from sklearn import ensemble
from itertools import chain
from typing import TextIO
import re

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import PredefinedSplit

from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix


# In[2]:


# data
genes = ['RPS14', 'CDC5L', 'POLR2I', 'RPS7', 'XAB2', 'RPS19BP1', 'RPL23A', 'SUPT6H', 'PRPF31', 'U2AF1', 'PSMD7',
         'Hsp10', 'RPS13', 'PHB', 'RPS9', 'EIF5B', 'RPS6', 'RPS11', 'SUPT5H', 'SNRPD2', 'RPL37', 'RPSA', 'COPS6',
         'DDX51', 'EIF4A3', 'KARS', 'RPL5', 'RPL32', 'SF3A1', 'RPS3A', 'SF3B3', 'POLR2D', 'RPS15A', 'RPL31', 'PRPF19',
         'SF3B2', 'RPS4X', 'CSE1L', 'RPL6', 'COPZ1', 'PSMB2', 'RPL7', 'PHB2', 'ARCN1', 'RPA2', 'NUP98', 'RPS3', 'EEF2',
         'USP39', 'PSMD1', 'NUP93', 'AQR', 'RPL34', 'PSMA1', 'RPS27A']


genes_filter_1 = ['RPS6', 'PRPF19', 'RPL34', 'Hsp10', 'POLR2I', 'EIF5B', 'RPL31',
       'RPS3A', 'CSE1L', 'XAB2', 'PSMD7', 'SUPT6H', 'EEF2', 'RPS11',
       'SNRPD2', 'RPL37', 'SF3B3', 'DDX51', 'RPL7', 'RPS9', 'KARS',
       'SF3A1', 'RPL32', 'PSMB2', 'RPS7', 'EIF4A3', 'U2AF1', 'PSMA1',
       'PHB', 'POLR2D', 'RPSA', 'RPL23A', 'NUP93', 'AQR', 'RPA2',
       'SUPT5H', 'RPL6', 'RPS13', 'SF3B2', 'RPS27A', 'PRPF31', 'COPZ1',
       'RPS4X', 'PSMD1', 'RPS14', 'NUP98', 'USP39', 'CDC5L', 'RPL5',
       'PHB2', 'RPS15A', 'RPS3', 'ARCN1', 'COPS6']

gene_split_index = {}
for i in range(len(genes_filter_1)):
    gene = genes_filter_1[i]
    gene_split_index[gene]= math.floor(i/6)


base_positions = {
    'A': 0,
    'T': 1,
    'C': 2,
    'G': 3,
    0: 'A',
    1: 'T',
    2: 'C',
    3: 'G',
}


# In[3]:

def create_gene_splits_filter1_kfold(gene_strings, values_to_split: list, kfold, split):
    # use number [0, 1, 2, 3, 4,...] as index
    genes_filter_1 = ['RPS6', 'PRPF19', 'RPL34', 'Hsp10', 'POLR2I', 'EIF5B', 'RPL31',
       'RPS3A', 'CSE1L', 'XAB2', 'PSMD7', 'SUPT6H', 'EEF2', 'RPS11',
       'SNRPD2', 'RPL37', 'SF3B3', 'DDX51', 'RPL7', 'RPS9', 'KARS',
       'SF3A1', 'RPL32', 'PSMB2', 'RPS7', 'EIF4A3', 'U2AF1', 'PSMA1',
       'PHB', 'POLR2D', 'RPSA', 'RPL23A', 'NUP93', 'AQR', 'RPA2',
       'SUPT5H', 'RPL6', 'RPS13', 'SF3B2', 'RPS27A', 'PRPF31', 'COPZ1',
       'RPS4X', 'PSMD1', 'RPS14', 'NUP98', 'USP39', 'CDC5L', 'RPL5',
       'PHB2', 'RPS15A', 'RPS3', 'ARCN1', 'COPS6']
    assert split >= 0 and split < kfold
    if kfold == 9:
        val_genes = genes_filter_1[split * 6: (split + 1) * 6]
        if split != 8:
            test_genes = genes_filter_1[((split + 1) * 6): (split + 2) * 6]
        else:
            test_genes = genes_filter_1[0:6]
    print('val:', val_genes)
    print('test:', test_genes)

    val_ids = list(chain(*[np.where(gene_strings == g)[0] for g in val_genes]))
    test_ids = list(chain(*[np.where(gene_strings == g)[0] for g in test_genes]))
    train_ids = list((set(range(len(gene_strings))) - set(val_ids) - set(test_ids)))

    train = [[arr[i] for i in train_ids] for arr in values_to_split]
    val = [[arr[i] for i in val_ids] for arr in values_to_split]
    test = [[arr[i] for i in test_ids] for arr in values_to_split]

    return train, val, test


def normalize(a: np.ndarray):
    """
    :param a: numpy array of size N x D, where N is number of examples, D is number of features
    :return: a, normalized so that all feature columns are now between 0 and 1
    """
    a_normed, norms = sklearn.preprocessing.normalize(a, norm='max', axis=0, return_norm=True)
    print("Norms:", norms)
    return a_normed

def one_hot_encode_sequence(seq, pad_to_len=-1):
    output_len = len(seq)
    if pad_to_len > 0:
        assert pad_to_len >= output_len
        output_len = pad_to_len

    encoded_seq = np.zeros((output_len, 4), dtype=np.float32)
    for i, base in enumerate(seq):
        encoded_seq[i][base_positions[base]] = 1
    return encoded_seq



dataset_filtered_csv_path = '../../../data/integrated_guide_feature_filtered_f24_mismatch3_all_flanks.csv'


#dataset
dataframe = pd.read_csv(dataset_filtered_csv_path)
dataframe = dataframe[dataframe['gene'].isin(genes_filter_1)] #filter out 1 gene

num_examples = len(dataframe['gene'].values)

encoded_guides = [one_hot_encode_sequence(guide).flatten() for guide in dataframe['guide'].values]



#classification
#classes = dataframe['binary_relative_ratio'].values
classes = dataframe['binary_relative_ratio_075f'].values
#classes = dataframe['top 20 pct per gene'].values
outputs = classes.astype(np.float32)
    
#guides_with_linfold = [np.concatenate((guide, linfold), axis=1) for guide, linfold in zip(encoded_guides, encoded_linearfold)]

all_cols = [encoded_guides, outputs]
    
#tr, val, te = create_gene_splits_kfold(dataframe['gene'].values, all_cols, 11, 4)

# group label to split
groups = dataframe['gene'].values 
# predefined split index
for g in gene_split_index.keys():
    dataframe.loc[dataframe['gene']== g,'predefined split index']= gene_split_index[g]
ps = PredefinedSplit(dataframe['predefined split index'].values)
print(ps.get_n_splits())


# hp tuning

clf = RandomForestClassifier(random_state=0)
grid = {'n_estimators':[100,200,400,800,1000,1200,1500,1800,2000],'max_features':['auto','sqrt','log2']}

#predefined splits
#gs = GridSearchCV(clf, grid, cv=ps.split(),scoring='accuracy')
gs = GridSearchCV(clf, grid, cv=ps.split(),scoring=['roc_auc','average_precision'],refit='roc_auc')

gs.fit(all_cols[0], all_cols[1])
print(gs.best_score_) #best cv score
print(gs.best_params_)
df_gridsearch = pd.DataFrame(gs.cv_results_)


df_gridsearch.to_csv('model_hp_results/randomforest_seqonly_hp.csv')


