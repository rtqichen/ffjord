from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib
# noninteractive background
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_training_curve(train_loss, validation_loss, fname='training_curve.pdf', labels=None):
    """
    Plots train_loss and validation loss as a function of optimization iteration
    :param train_loss: np.array of train_loss (1D or 2D)
    :param validation_loss: np.array of validation loss (1D or 2D)
    :param fname: output file name
    :param labels: if train_loss and validation loss are 2D, then labels indicate which variable is varied
    accross training curves.
    :return: None
    """

    plt.close()

    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    if len(train_loss.shape) == 1:
        # Single training curve
        fig, ax = plt.subplots(nrows=1, ncols=1)
        figsize = (6, 4)

        if train_loss.shape[0] == validation_loss.shape[0]:
            # validation score evaluated every iteration
            x = np.arange(train_loss.shape[0])
            ax.plot(x, train_loss, '-', lw=2., color='black', label='train')
            ax.plot(x, validation_loss, '-', lw=2., color='blue', label='val')

        elif train_loss.shape[0] % validation_loss.shape[0] == 0:
            # validation score evaluated every epoch
            x = np.arange(train_loss.shape[0])
            ax.plot(x, train_loss, '-', lw=2., color='black', label='train')

            x = np.arange(validation_loss.shape[0])
            x = (x + 1) * train_loss.shape[0] / validation_loss.shape[0]
            ax.plot(x, validation_loss, '-', lw=2., color='blue', label='val')
        else:
            raise ValueError('Length of train_loss and validation_loss must be equal or divisible')

        miny = np.minimum(validation_loss.min(), train_loss.min()) - 20.
        maxy = np.maximum(validation_loss.max(), train_loss.max()) + 30.
        ax.set_ylim([miny, maxy])

    elif len(train_loss.shape) == 2:
        # Multiple training curves

        cmap = plt.cm.brg

        cNorm = matplotlib.colors.Normalize(vmin=0, vmax=train_loss.shape[0])
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        figsize = (6, 4)

        if labels is None:
            labels = ['%d' % i for i in range(train_loss.shape[0])]

        if train_loss.shape[1] == validation_loss.shape[1]:
            for i in range(train_loss.shape[0]):
                color_val = scalarMap.to_rgba(i)

                # validation score evaluated every iteration
                x = np.arange(train_loss.shape[0])
                ax.plot(x, train_loss[i], '-', lw=2., color=color_val, label=labels[i])
                ax.plot(x, validation_loss[i], '--', lw=2., color=color_val)

        elif train_loss.shape[1] % validation_loss.shape[1] == 0:
            for i in range(train_loss.shape[0]):
                color_val = scalarMap.to_rgba(i)

                # validation score evaluated every epoch
                x = np.arange(train_loss.shape[1])
                ax.plot(x, train_loss[i], '-', lw=2., color=color_val, label=labels[i])

                x = np.arange(validation_loss.shape[1])
                x = (x + 1) * train_loss.shape[1] / validation_loss.shape[1]
                ax.plot(x, validation_loss[i], '-', lw=2., color=color_val)

        miny = np.minimum(validation_loss.min(), train_loss.min()) - 20.
        maxy = np.maximum(validation_loss.max(), train_loss.max()) + 30.
        ax.set_ylim([miny, maxy])

    else:
        raise ValueError('train_loss and validation_loss must be 1D or 2D arrays')

    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    plt.title('Training and validation loss')

    fig.set_size_inches(figsize)
    fig.subplots_adjust(hspace=0.1)
    plt.savefig(fname, bbox_inches='tight')

    plt.close()
