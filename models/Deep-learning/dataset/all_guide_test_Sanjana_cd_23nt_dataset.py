import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random

from dataset.dataset_utils import *


def all_guide_test_Sanjana_cd_23nt_dataset(args):
    dataframe = pd.read_csv('../../data/integrated_guide_feature_filtered_f24_mismatch3_all_features.csv')
    num_examples = len(dataframe['gene'].values)

    #lin_seq_dict, lin_result_dict = parse_guide_linearfold_fasta_into_dict_contrafold()

    #encoded_guides = [one_hot_encode_sequence(guide) for guide in dataframe['guide'].values]
    encoded_guides = [one_hot_encode_sequence(guide)[:args.guidelength] for guide in dataframe['guide'].values]
    #encoded_linearfold = [one_hot_encode_linearfold(lin_seq_dict[guide], remove_universal_start=True) for guide in
    #                      dataframe['guide'].values]

    
    other_single_value_inputs = np.empty((8, num_examples))
    other_single_value_inputs[0, :] = dataframe['linearfold_vals'].values
    #other_single_value_inputs[1, :] = linearfold_dr
    other_single_value_inputs[1, :] = dataframe['is_5UTR'].values
    other_single_value_inputs[2, :] = dataframe['is_CDS'].values
    other_single_value_inputs[3, :] = dataframe['is_3UTR'].values
    #other_single_value_inputs[4, :] = dataframe['refseq_target_transcript_percent'].values
    other_single_value_inputs[4, :] = dataframe['target unfold energy']
    other_single_value_inputs[5, :] = dataframe['UTR5_position'].values
    other_single_value_inputs[6, :] = dataframe['CDS_position'].values
    other_single_value_inputs[7, :] = dataframe['UTR3_position'].values


    #classes = 1- dataframe['old_binary_relative_ratio_gene20'].values
    classes = dataframe['binary_relative_ratio_075f'].values
    outputs = dataframe['relative_ratio'].values if args.regression else classes.astype(np.float32)
    #outputs = outputs.tolist()


    all_cols = [encoded_guides,  # will be N x 4 from guide encoding
                normalize(other_single_value_inputs.T),
                # classes,
                outputs
                ]

    #tr = all_cols

    if args.kfold == None:
        tr, val, te_train = create_gene_splits(dataframe['gene'].values, all_cols)
    else:
        #tr, val = create_gene_splits_no_test_kfold(dataframe['gene'].values, all_cols, args.kfold, args.split)
        #tr, val, te_train = create_gene_splits_kfold(dataframe['gene'].values, all_cols, args.kfold, args.split)
        #tr, val, te_train = create_gene_splits_filter1_kfold(dataframe['gene'].values, all_cols, args.kfold, args.split)
        #tr, val = create_gene_splits_filter1_no_test_kfold(dataframe['gene'].values, all_cols, args.kfold, args.split)
        tr, val = create_gene_splits_filter1_test_asval_kfold(dataframe['gene'].values, all_cols, args.kfold, args.split)

    # test set data, nbt 23nt
    tedf = pd.read_csv('dataset/Sanjana_CD_data_23nt_filtered_all_features.csv')
    encoded_guides_te = [one_hot_encode_sequence(guide) for guide in tedf['guide'].values]
    num_examples_te = len(tedf['guide'].values)

    
    outputs_te = tedf['relative_ratio'].values if args.regression else tedf['binary_relative_ratio'].values
    #outputs_te =outputs_te.tolist()


    other_single_value_inputs_te = np.empty((8, num_examples_te))
    other_single_value_inputs_te[0, :] = tedf['linearfold_vals'].values/max(dataframe['linearfold_vals'].values) #normalize as the training data
    other_single_value_inputs_te[1, :] = tedf['is_5UTR'].values
    other_single_value_inputs_te[2, :] = tedf['is_CDS'].values
    other_single_value_inputs_te[3, :] = tedf['is_3UTR'].values
    #other_single_value_inputs_te[4, :] = tedf['refseq_target_transcript_percent'].values
    other_single_value_inputs_te[4, :] = tedf['target unfold energy']/max(dataframe['target unfold energy'].values)
    other_single_value_inputs_te[5, :] = tedf['UTR5_position']/max(dataframe['UTR5_position'].values)
    other_single_value_inputs_te[6, :] = tedf['CDS_position']/max(dataframe['CDS_position'].values)
    other_single_value_inputs_te[7, :] = tedf['UTR3_position']/max(dataframe['UTR3_position'].values)


    all_cols_te = [encoded_guides_te,  # will be N x 4 from guide encoding
                other_single_value_inputs_te.T,
                outputs_te
                ]

    te = all_cols_te    

    tr_out = tr[-1]
    tr = tuple(tr[:-1])
    val_out = val[-1]
    val = tuple(val[:-1])
    te_out = te[-1]
    te = tuple(te[:-1])

    train_dataset = tf.data.Dataset.from_tensor_slices((tr, tr_out))
    val_dataset = tf.data.Dataset.from_tensor_slices((val, val_out))
    test_dataset = tf.data.Dataset.from_tensor_slices((te, te_out))

    # shuffle and batch
    train_dataset = prep_dataset(train_dataset, batch_size=128)
    val_dataset = prep_dataset(val_dataset, batch_size=128)
    test_dataset = prep_dataset(test_dataset, batch_size=128)

    return train_dataset, val_dataset, test_dataset
