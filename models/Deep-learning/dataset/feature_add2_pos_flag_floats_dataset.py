import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random

#from dataset.dataset_filtered_utils import *
from dataset.dataset_utils import *
#from dataset.dataset_utils import normalize


def feature_add2_pos_flag_floats_dataset(args):
    dataframe = pd.read_csv('dataset/integrated_guide_feature_filtered_f24_mismatch3_all_flanks.csv')
    genes_filter_1 = ['RPS6', 'PRPF19', 'RPL34', 'Hsp10', 'POLR2I', 'EIF5B', 'RPL31',
       'RPS3A', 'CSE1L', 'XAB2', 'PSMD7', 'SUPT6H', 'EEF2', 'RPS11',
       'SNRPD2', 'RPL37', 'SF3B3', 'DDX51', 'RPL7', 'RPS9', 'KARS',
       'SF3A1', 'RPL32', 'PSMB2', 'RPS7', 'EIF4A3', 'U2AF1', 'PSMA1',
       'PHB', 'POLR2D', 'RPSA', 'RPL23A', 'NUP93', 'AQR', 'RPA2',
       'SUPT5H', 'RPL6', 'RPS13', 'SF3B2', 'RPS27A', 'PRPF31', 'COPZ1',
       'RPS4X', 'PSMD1', 'RPS14', 'NUP98', 'USP39', 'CDC5L', 'RPL5',
       'PHB2', 'RPS15A', 'RPS3', 'ARCN1', 'COPS6']
    dataframe = dataframe[dataframe['gene'].isin(genes_filter_1)] #filter out 1 gene

    num_examples = len(dataframe['gene'].values)

    #lin_seq_dict, lin_result_dict = parse_guide_linearfold_fasta_into_dict()
    lin_seq_dict, lin_result_dict = parse_guide_linearfold_fasta_into_dict_contrafold()

    encoded_guides = [one_hot_encode_sequence(guide) for guide in dataframe['guide'].values]
    encoded_linearfold = [one_hot_encode_linearfold(lin_seq_dict[guide], remove_universal_start=True) for guide in
                          dataframe['guide'].values]

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

    #rnafe = dataframe['contrafold_2'].values
    #rnafe_rel = rnafe - rnafe.min()

    # encoded_guides = [reverse_complement_encoding(guide) for guide in dataframe['guide'].values]

    #target with nearby seq, dg of native and unfolded
    flank_l = int(args.flanklength)
    lin_seq_flanks_dict, lin_result_flanks_dict = parse_target_flanks_linearfold_fasta_into_dict_contrafold(flank_len = flank_l)
    linearfold_vals_target = [lin_result_flanks_dict[target_flanks] for target_flanks in dataframe['nearby_seq_all_'+str(flank_l)].values] #native energy
    #lin_seq_flanks = [lin_seq_flanks_dict[target_flanks] for target_flanks in dataframe['nearby_seq_all_100'].values]
    unfold_lin_seq_flanks_dict, unfold_lin_result_flanks_dict = parse_target_flanks_constraints_linearfold_fasta_into_dict_contrafold(flank_len = flank_l)
    unfold_linearfold_vals_target = [unfold_lin_result_flanks_dict[target_flanks] for target_flanks in dataframe['nearby_seq_all_'+str(flank_l)].values] #unfolded target energy
    ddg = [] #energy required to unfold the guide binding region
    for jj in range(num_examples):
        ddg.append((linearfold_vals_target[jj]-unfold_linearfold_vals_target[jj]))


    #classes = 1 - dataframe['old_binary_relative_ratio_gene20'].values
    #classes = dataframe['binary_relative_ratio'].values
    classes = dataframe['binary_relative_ratio_075f'].values


    outputs = dataframe['relative_ratio'].values if args.regression else classes.astype(np.float32)

    other_single_value_inputs = np.empty((6, num_examples))
    #other_single_value_inputs[0, :] = linearfold_vals
    #other_single_value_inputs[1, :] = linearfold_dr
    other_single_value_inputs[0, :] = dataframe['is_5UTR'].values
    other_single_value_inputs[1, :] = dataframe['is_CDS'].values
    other_single_value_inputs[2, :] = dataframe['is_3UTR'].values
    #other_single_value_inputs[4, :] = dataframe['refseq_target_transcript_percent'].values
    #other_single_value_inputs[1, :] = ddg
    other_single_value_inputs[3, :] = dataframe['UTR5_position'].values
    other_single_value_inputs[4, :] = dataframe['CDS_position'].values
    other_single_value_inputs[5, :] = dataframe['UTR3_position'].values


    all_cols = [encoded_guides,
                normalize(other_single_value_inputs.T),
                # classes,
                outputs
                ]

    # tr, val, te = create_gene_splits(dataframe['gene'].values, all_cols)
    #tr, val, te = create_gene_splits_kfold(dataframe['gene'].values, all_cols, 0)
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
