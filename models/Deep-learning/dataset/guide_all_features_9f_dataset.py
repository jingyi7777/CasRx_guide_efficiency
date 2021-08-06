import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random

#from dataset.dataset_filtered_utils import *
from dataset.dataset_utils import *
#from dataset.dataset_utils import normalize


def guide_all_features_9f_dataset(args):
    dataframe = pd.read_csv('../../data/integrated_guide_feature_filtered_f24_mismatch3_all_features.csv')
    num_examples = len(dataframe['gene'].values)

    #lin_seq_dict, lin_result_dict = parse_guide_linearfold_fasta_into_dict()
    #lin_seq_dict, lin_result_dict = parse_guide_linearfold_fasta_into_dict_contrafold()

    encoded_guides = [one_hot_encode_sequence(guide) for guide in dataframe['guide'].values]
    #encoded_linearfold = [one_hot_encode_linearfold(lin_seq_dict[guide], remove_universal_start=True) for guide in
    #                      dataframe['guide'].values]

    # encoded_guides = [reverse_complement_encoding(guide) for guide in dataframe['guide'].values]

    #target with nearby seq, dg of native and unfolded
    # flank_l = int(args.flanklength)
    # lin_seq_flanks_dict, lin_result_flanks_dict = parse_target_flanks_linearfold_fasta_into_dict_contrafold(flank_len = flank_l)
    # linearfold_vals_target = [lin_result_flanks_dict[target_flanks] for target_flanks in dataframe['nearby_seq_all_'+str(flank_l)].values] #native energy
    # #lin_seq_flanks = [lin_seq_flanks_dict[target_flanks] for target_flanks in dataframe['nearby_seq_all_100'].values]
    # unfold_lin_seq_flanks_dict, unfold_lin_result_flanks_dict = parse_target_flanks_constraints_linearfold_fasta_into_dict_contrafold(flank_len = flank_l)
    # unfold_linearfold_vals_target = [unfold_lin_result_flanks_dict[target_flanks] for target_flanks in dataframe['nearby_seq_all_'+str(flank_l)].values] #unfolded target energy
    # ddg = [] #energy required to unfold the guide binding region
    # for jj in range(num_examples):
    #     ddg.append((linearfold_vals_target[jj]-unfold_linearfold_vals_target[jj]))

    classes = dataframe['binary_relative_ratio_075f'].values
    outputs = dataframe['relative_ratio'].values if args.regression else classes.astype(np.float32)

    other_single_value_inputs = np.empty((9, num_examples))
    other_single_value_inputs[0, :] = dataframe['linearfold_vals'].values
    other_single_value_inputs[1, :] = dataframe['is_5UTR'].values
    other_single_value_inputs[2, :] = dataframe['is_CDS'].values
    other_single_value_inputs[3, :] = dataframe['is_3UTR'].values
    other_single_value_inputs[4, :] = dataframe['refseq_target_transcript_percent'].values
    other_single_value_inputs[5, :] = dataframe['target unfold energy'].values
    other_single_value_inputs[6, :] = dataframe['UTR5_position'].values
    other_single_value_inputs[7, :] = dataframe['CDS_position'].values
    other_single_value_inputs[8, :] = dataframe['UTR3_position'].values
    #other_single_value_inputs[9, :] = dataframe['linearfold_dr_flag'].values
    #other_single_value_inputs[10, :] = dataframe['GC_content'].values


    all_cols = [encoded_guides,
                normalize(other_single_value_inputs.T),
                # classes,
                outputs
                ]


    if args.kfold == None:
        tr, val, te = create_gene_splits(dataframe['gene'].values, all_cols)
    else:
        #tr, val, te = create_gene_splits_kfold(dataframe['gene'].values, all_cols, args.kfold, args.split)
        tr, val, te = create_gene_splits_filter1_kfold(dataframe['gene'].values, all_cols, args.kfold, args.split)

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
