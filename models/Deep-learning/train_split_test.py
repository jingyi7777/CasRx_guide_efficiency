import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import datetime
import pdb
import numpy as np
from dataset import find_dataset_generator_using_name
from models import find_model_using_name
from options.options import get_arguments
from matplotlib import pyplot as plt
from utils import *

tf.random.set_seed(0)
#random.seed(0)
np.random.seed(0)

def logits_mean_absolute_error(y_true, y_pred):
    y_pred = tf.sigmoid(y_pred)
    return keras.metrics.mean_absolute_error(y_true, y_pred)


def logits_mean_squared_error(y_true, y_pred):
    y_pred = tf.sigmoid(y_pred)
    return keras.metrics.mean_squared_error(y_true, y_pred)

def wbce(y_true, y_pred, weight1 = 1, weight0 = 1) :
    y_true = tf.keras.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = tf.keras.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0 )
    return tf.keras.mean( logloss, axis=-1)

def train_split_test(args):
    model_generator = find_model_using_name(args.model)
    dataset_generator = find_dataset_generator_using_name(args.dataset)

    model = model_generator(args)
    model: keras.Model
    train_dataset, val_dataset, test_dataset = dataset_generator(args)
    unshuffled_train = train_dataset
    train_dataset = train_dataset.shuffle(len(train_dataset), reshuffle_each_iteration=True)
    if args.saved != None:
        if args.regression:
            model = keras.models.load_model(args.saved,custom_objects={'logits_mean_absolute_error':logits_mean_absolute_error,'logits_mean_squared_error':logits_mean_squared_error})
        else:
            model = keras.models.load_model(args.saved)
        #model = keras.models.load_model(args.saved)
    else:
        if args.focal:
            loss = lambda y_true, y_pred: tfa.losses.sigmoid_focal_crossentropy(y_true, y_pred, from_logits=True)
        else:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        optimizer = keras.optimizers.Adam(lr=args.lr)

        if args.regression:
            metrics = [logits_mean_absolute_error, logits_mean_squared_error]
            model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=metrics)
        else:
            # metrics = ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()] #doesn't work with logits
            metrics = ['accuracy']
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        unique_train_signature = '%s/%s/%s/%s' % (
            dataset_generator.__name__,
            model_generator.__name__,
            'regression' if args.regression else 'classification',
            ('fold_'+ str(args.split))
            #datetime.datetime.now().isoformat()
        )

        callbacks = [
            keras.callbacks.EarlyStopping(patience=16, restore_best_weights=True),
            keras.callbacks.TensorBoard(log_dir='logs/%s' % unique_train_signature,
                                        # update_freq=100
                                        ),
            keras.callbacks.ModelCheckpoint('checkpoints/%s' % unique_train_signature, save_best_only=True)
        ]
        if args.weighted:
            #weight_zeros = 1.0 / (0.8 * 2.0) # 1 / (% of class * number of classes)
            #weight_ones  = 1.0 / (0.2 * 2.0)

            weight_zeros = 1.0 
            weight_ones  = 2.0
            class_weights = {0: weight_zeros, 1: weight_ones}

            history = model.fit(train_dataset, epochs=200,
                            validation_data=val_dataset,
                            verbose=1,
                            callbacks=callbacks,
                            class_weight=class_weights
                            )
        else:
            history = model.fit(train_dataset, epochs=200,
                            validation_data=val_dataset,
                            verbose=1,
                            callbacks=callbacks
                            )
        model.save('saved_model/%s' % unique_train_signature)
        
    if args.regression:
        #get_regression_metrics_nbt(model, test_dataset, args.regression, args.kfold, args.split, model_name=args.model, dataset_name=args.dataset)
        get_pseudo_roc_for_regression(model, test_dataset, args.regression, args.kfold, args.split, model_name=args.model, dataset_name=args.dataset)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        get_classification_metrics(model, unshuffled_train, fig, ax1, ax2, args.regression, args.kfold, args.split, args.guidelength, model_name=args.model + ' on train',
                                   dataset_name=args.dataset, save=False)
        get_classification_metrics(model, val_dataset, fig, ax1, ax2, args.regression, args.kfold, args.split, args.guidelength, model_name=args.model + ' on val', dataset_name=args.dataset, 
                                   save=False)
        get_classification_metrics_test(args.testset_path, model, test_dataset, fig, ax1, ax2, args.regression, args.kfold, args.split, args.guidelength, model_name=args.model, dataset_name=args.dataset)

    if args.gradients:
        integrated_gradients(model, test_dataset, args.regression, args.kfold, args.split, model_name=args.model, dataset_name=args.dataset)

    print("done!")


if __name__ == '__main__':
    # Enable the following 3 lines if using a graphics card and you get CUDNN_STATUS_INTERNAL_ERROR
    args = get_arguments()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        gpus_to_use = args.gpu_ids.split(",")
        for i in range(len(gpus_to_use)):
            gpu_id = int(gpus_to_use[i])
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)

    train_split_test(args)
