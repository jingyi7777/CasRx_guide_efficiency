from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from kerastuner import Hyperband
from matplotlib import pyplot as plt
import datetime
import pdb

from dataset import find_dataset_generator_using_name
from models import find_hp_model_using_name
from options.options import get_arguments
from utils import get_classification_metrics, get_pseudo_roc_for_regression


def hyperparam_tune(args):
    model_generator = find_hp_model_using_name(args.model)
    dataset_generator = find_dataset_generator_using_name(args.dataset)

    train_dataset, val_dataset, test_dataset = dataset_generator(args)

    unshuffled_train = train_dataset
    train_dataset = train_dataset.shuffle(len(train_dataset), reshuffle_each_iteration=True)

    tuner = Hyperband(
        model_generator,
        'val_loss',
        max_epochs=100,
        # hyperband_iterations=5,
        directory='hyper_tune/%s/%s' % (dataset_generator.__name__, model_generator.__name__),
        overwrite=True,
    )

    # tuner.reload()

    unique_train_signature = '%s/%s/%s/%s' % (
        dataset_generator.__name__,
        model_generator.__name__,
        'regression' if args.regression else 'classification',
        datetime.datetime.now().isoformat()
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=16, restore_best_weights=True),
        keras.callbacks.TensorBoard(log_dir='logs/%s' % unique_train_signature, write_graph=True, write_images=True, write_grads=True
                                    # update_freq=100
                                    ),
        keras.callbacks.ModelCheckpoint('checkpoints/%s' % unique_train_signature, save_best_only=True)
    ]

    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        verbose=1,
        callbacks=callbacks
    )
    tuner.results_summary()
    best = tuner.get_best_models(1)

    #for i, model in enumerate(best):
    #    model: keras.Model
    #    fig, (ax1, ax2) = plt.subplots(1, 2)
    #    get_classification_metrics(model, unshuffled_train, fig, ax1, ax2, args.regression, args.kfold, args.split, model_name=args.model + ' on train',
    #                               dataset_name=args.dataset, save=False)
    #    get_classification_metrics(model, val_dataset, fig, ax1, ax2, args.regression, args.kfold, args.split, model_name=args.model + ' on val', dataset_name=args.dataset, 
    #                               save=False)
    #    get_classification_metrics(model, test_dataset, fig, ax1, ax2, args.regression, args.kfold, args.split, model_name=args.model, dataset_name=args.dataset)
        # if i == 1:
        #     model.save('best_model_no_gene_auc_roc')
    print("done!")


if __name__ == '__main__':
    # Enable the following 3 lines if using a graphics card and you get CUDNN_STATUS_INTERNAL_ERROR
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    args = get_arguments()
    hyperparam_tune(args)
