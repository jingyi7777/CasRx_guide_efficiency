import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random

#from dataset.dataset_filtered_utils import *
from dataset.dataset_utils import *
#from dataset.dataset_utils import normalize


def feature_sub_guideonly_flanks_dataset(args):
    dataframe = pd.read_csv('../../data/integrated_guide_feature_filtered_f24_mismatch3_all_features.csv')
    num_examples = len(dataframe['gene'].values)

    #encoded_guides = [one_hot_encode_sequence(guide)[:args.guidelength] for guide in dataframe['guide'].values]

    flank_l = int(args.flanklength)
    encoded_guides = [reverse_complement_encoding(guide) for guide in dataframe['guide'].values]
    gene_encoding_dict = get_gene_encodings(pad=False)
    encoded_genes = [gene_encoding_dict[gene] for gene in dataframe['gene'].values]
    positions = dataframe['pos'].values

    guide_flanks = [get_nearby_encoding_rv(guide, loc, gene, num_bases_either_side=flank_l) for guide, loc, gene in
                   zip(encoded_guides, positions, encoded_genes)]
                 
    classes = dataframe['binary_relative_ratio_075f'].values

    outputs = dataframe['relative_ratio'].values if args.regression else classes.astype(np.float32)

    all_cols = [guide_flanks,
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
