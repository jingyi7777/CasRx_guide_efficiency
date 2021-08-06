from matplotlib import pyplot as plt


def plot_metrics(model_history, metrics=('loss')):

    for metric in metrics:
        # summarize history for each metric
        plt.plot(model_history[metric])
        plt.plot(model_history['val_%s' % metric])
        plt.title('model %s' % metric)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
