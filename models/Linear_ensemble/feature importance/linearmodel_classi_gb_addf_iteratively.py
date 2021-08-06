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
from sklearn.model_selection import cross_validate

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



#distinguishing nmer data
#nmer csv
# enrichf = '../nmer_results/detailed_nmer_enriched_allfold_stats.csv'
# depletef = '../nmer_results/detailed_nmer_depleted_allfold_stats.csv'
# df_enriched = pd.read_csv(enrichf)
# df_depleted = pd.read_csv(depletef)
# enrichnmer = list(df_enriched['nmer'].values)
# depltednmer = list(df_depleted['nmer'].values)

# # 4mer
# dslist=[]
# for i in enrichnmer:
#     if len(i)==4:
#         dslist.append(i)
# for j in depltednmer:
#     if len(j)==4:
#         dslist.append(j)
#print(dslist)



def create_gene_splits_kfold(gene_strings, values_to_split: list, kfold, split):
    # use number [0, 1, 2, 3, 4] as index
    assert split >= 0 and split < kfold
    if kfold == 5:
        non_train_genes = genes[split * 11: (split + 1) * 11]
        val_genes = non_train_genes[:5]
        test_genes = non_train_genes[5:]
    elif kfold == 11:
        num_genes = len(genes)
        val_genes = genes[split * 5: (split + 1) * 5]
        if split != 10:
            test_genes = genes[((split + 1) * 5): (split + 2) * 5]
        else:
            test_genes = genes[0:5]
    print('val:', val_genes)
    print('test:', test_genes)

    val_ids = list(chain(*[np.where(gene_strings == g)[0] for g in val_genes]))
    test_ids = list(chain(*[np.where(gene_strings == g)[0] for g in test_genes]))
    train_ids = list((set(range(len(gene_strings))) - set(val_ids) - set(test_ids)))

    train = [[arr[i] for i in train_ids] for arr in values_to_split]
    val = [[arr[i] for i in val_ids] for arr in values_to_split]
    test = [[arr[i] for i in test_ids] for arr in values_to_split]

    return train, val, test

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

def parse_guide_linearfold_fasta_into_dict_contrafold():
    fasta_file = open(linearfold_fasta_path_c)
    seq_dict = {}
    score_dict = {}

    def parse_one_example(fasta: TextIO):
        descr_line = fasta.readline()
        if not descr_line:
            return None, None, None
        guide_seq = fasta.readline().strip()[36:]
        linearfold_and_result = fasta.readline()
        match = re.match('([\\.|\\(|\\)]+) \((\-?[0-9]*\.[0-9]+)\)', linearfold_and_result)
        linseq, score = match.groups()
        score = float(score)

        assert '>' in descr_line

        return guide_seq, linseq, score

    while True:
        key, seq, score = parse_one_example(fasta_file)
        if key is None:
            break
        seq_dict[key] = seq
        score_dict[key] = score

    fasta_file.close()

    return seq_dict, score_dict


def parse_target_flanks_linearfold_fasta_into_dict_contrafold(flank_len = 15):
    flank_num = flank_len
    fname = '../dataset/linearfold_output/linfold_guides_nearby'+str(flank_num)+'_output.txt'
    fasta_file = open(fname)
    seq_dict = {}
    score_dict = {}

    def parse_one_example(fasta: TextIO):
        descr_line = fasta.readline()
        if not descr_line:
            return None, None, None
        target_seq = fasta.readline().strip() #target with flanks
        linearfold_and_result = fasta.readline()
        match = re.match('([\\.|\\(|\\)]+) \((\-?[0-9]*\.[0-9]+)\)', linearfold_and_result)
        linseq, score = match.groups()
        score = float(score)

        assert '>' in descr_line

        return target_seq, linseq, score #return target seq with flanks

    while True:
        key, seq, score = parse_one_example(fasta_file)
        if key is None:
            break
        seq_dict[key] = seq
        score_dict[key] = score

    fasta_file.close()

    return seq_dict, score_dict


def parse_target_flanks_constraints_linearfold_fasta_into_dict_contrafold(flank_len = 15):
    flank_num = flank_len
    fname = '../dataset/linearfold_output/linfold_guides_constrains_nearby'+str(flank_num)+'_output.txt'
    fasta_file = open(fname)
    seq_dict = {}
    score_dict = {}

    def parse_one_example(fasta: TextIO):
        descr_line = fasta.readline()
        if not descr_line:
            return None, None, None
        target_seq = fasta.readline().strip() #target with flanks
        constraints = fasta.readline()
        linearfold_and_result = fasta.readline()
        match = re.match('([\\.|\\(|\\)]+) \((\-?[0-9]*\.[0-9]+)\)', linearfold_and_result)
        linseq, score = match.groups()
        score = float(score)

        assert '>' in descr_line

        return target_seq, linseq, score #return target seq with flanks

    while True:
        key, seq, score = parse_one_example(fasta_file) #key is target seq with flanks
        if key is None:
            break
        seq_dict[key] = seq #linfold_seq
        score_dict[key] = score #linfold_score

    fasta_file.close()

    return seq_dict, score_dict


dataset_filtered_csv_path = '../dataset/integrated_guide_feature_filtered_f24_mismatch3_all_flanks.csv'
linearfold_fasta_path_c = '../dataset/guides_linearfold_c.txt'

#dataset
dataframe = pd.read_csv(dataset_filtered_csv_path)
dataframe = dataframe[dataframe['gene'].isin(genes_filter_1)] #filter out 1 gene

num_examples = len(dataframe['gene'].values)

#lin_seq_dict, lin_result_dict = parse_guide_linearfold_fasta_into_dict()
lin_seq_dict, lin_result_dict = parse_guide_linearfold_fasta_into_dict_contrafold()

encoded_guides = [one_hot_encode_sequence(guide).flatten() for guide in dataframe['guide'].values]
#encoded_linearfold = [one_hot_encode_linearfold(lin_seq_dict[guide], remove_universal_start=True) for guide in
#                      dataframe['guide'].values]

linearfold_dr = [lin_seq_dict[guide][0:36] for guide in dataframe['guide'].values]
ref_dr = '.....(((((((.(((....))).))))))).....'
dr_disr_num =0
for jj in range(num_examples):
    if linearfold_dr[jj] == ref_dr:
        linearfold_dr[jj] = 0
    else:
        linearfold_dr[jj] = 1
        dr_disr_num += 1
print('dr_disr_num:'+str(dr_disr_num))  


linearfold_vals = [lin_result_dict[guide] for guide in dataframe['guide'].values]
for ii in range(num_examples):
    linearfold_vals[ii] = abs(linearfold_vals[ii]-6.48)

#target with nearby seq, dg of native and unfolded
flank_l = 15
lin_seq_flanks_dict, lin_result_flanks_dict = parse_target_flanks_linearfold_fasta_into_dict_contrafold(flank_len = flank_l)
linearfold_vals_target = [lin_result_flanks_dict[target_flanks] for target_flanks in dataframe['nearby_seq_all_'+str(flank_l)].values] #native energy
unfold_lin_seq_flanks_dict, unfold_lin_result_flanks_dict = parse_target_flanks_constraints_linearfold_fasta_into_dict_contrafold(flank_len = flank_l)
unfold_linearfold_vals_target = [unfold_lin_result_flanks_dict[target_flanks] for target_flanks in dataframe['nearby_seq_all_'+str(flank_l)].values] #unfolded target energy
ddg = [] #energy required to unfold the guide binding region
for jj in range(num_examples):
    ddg.append((linearfold_vals_target[jj]-unfold_linearfold_vals_target[jj]))



        
#three pos floats classification, no nmer flags
other_single_value_inputs = np.empty((9, num_examples))
other_single_value_inputs[0, :] = linearfold_vals
#other_single_value_inputs[1, :] = linearfold_dr
other_single_value_inputs[1, :] = dataframe['is_5UTR'].values
other_single_value_inputs[2, :] = dataframe['is_CDS'].values
other_single_value_inputs[3, :] = dataframe['is_3UTR'].values
other_single_value_inputs[4, :] = dataframe['refseq_target_transcript_percent'].values
other_single_value_inputs[5, :] = ddg
other_single_value_inputs[6, :] = dataframe['UTR5_position'].values
other_single_value_inputs[7, :] = dataframe['CDS_position'].values
other_single_value_inputs[8, :] = dataframe['UTR3_position'].values

#classification
#classes = dataframe['binary_relative_ratio'].values
classes = dataframe['binary_relative_ratio_075f'].values
#classes = dataframe['top 20 pct per gene'].values
outputs = classes.astype(np.float32)
    
#guides_with_linfold = [np.concatenate((guide, linfold), axis=1) for guide, linfold in zip(encoded_guides, encoded_linearfold)]

all_cols = [np.concatenate((encoded_guides, normalize(other_single_value_inputs.T)),axis=1), outputs]
    
# group label to split
groups = dataframe['gene'].values 
# predefined split index
for g in gene_split_index.keys():
    dataframe.loc[dataframe['gene']== g,'predefined split index']= gene_split_index[g]
ps = PredefinedSplit(dataframe['predefined split index'].values)
print(ps.get_n_splits())


#feature labels
nuc_labels = []
for p in range(30):
    for bi in range(4):
        nuc_label = 'pos'+str(p)+'_'+ base_positions[bi]
        nuc_labels.append(nuc_label)
        
#nuc_labels = ["guide_pos" + str(i) for i in range(120)]
feature_list=['linearfold_vals','is_5UTR','is_CDS','is_3UTR','target_transcript_percent','target unfold energy',
'UTR5_position','CDS_position','UTR3_position']
feature_names = nuc_labels+feature_list

df_select = pd.DataFrame(data=all_cols[0],
                         columns=feature_names)
df_select['output label']=all_cols[1]


#GradientBoostingClassifier


#dataset_lofo = Dataset(df=df_select, target='output label', features=[col for col in df_select.columns if col != target])

# define GradientBoostingClassifier
clf = ensemble.GradientBoostingClassifier(random_state=0,max_depth=4,
                                         max_features='sqrt', n_estimators=2000)

#guide seq only model scores
y = df_select['output label'].values
x_seq = df_select[nuc_labels].values
cv_results_base = cross_validate(clf,x_seq, y, cv=ps.split(), scoring=['roc_auc','average_precision'])

df_auroc = pd.DataFrame()
df_auroc['seq only']=cv_results_base['test_roc_auc']
df_auprc = pd.DataFrame()
df_auprc['seq only']=cv_results_base['test_average_precision']


feature_list_ordered =[['UTR5_position','CDS_position','UTR3_position'],['is_5UTR','is_CDS','is_3UTR'],'target unfold energy','linearfold_vals','target_transcript_percent']

feature_add = nuc_labels
for f in feature_list_ordered: #secondary feature addition
	if isinstance(f, list): #whether f is grouped features
		feature_add = feature_add + f
	else:
		feature_add = feature_add +[f]

	X = df_select[feature_add].values
	#y = df_select['output label'].values
	cv_results = cross_validate(clf, X, y, cv=ps.split(), scoring=['roc_auc','average_precision'])
	if isinstance(f, list): #whether f is grouped features
		f = f[0]+'_three'
	df_auroc[f]=cv_results['test_roc_auc']
	df_auprc[f]=cv_results['test_average_precision']
	

df_auroc.to_csv('linearmodel_classi_gb_addf_iteratively_no_nmer_auroc.csv')
df_auprc.to_csv('linearmodel_classi_gb_addf_iteratively_no_nmer_auprc.csv')
