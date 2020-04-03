from __future__ import absolute_import, division, print_function, unicode_literals

# to enforce static type in functions
from typing import List

import tensorflow as tf
import numpy as np
import imageio
import pandas as pd
# for plotting
import matplotlib.pyplot as plt
# for logging
from utils.logger import get_log_object

log = get_log_object()

    #plt.show()


def plot_graphs(history: tf.keras.callbacks.History,
                plot_title: str,
                string_metric: str,
                location_plot: str) -> None:

    """
    Function that generates plot of loss and accuracy vs epoch
    :param history: TF2 History object that comes from model fitting
    :param plot_title: string to be used as plot title
    :param string_metric: string that captures the metric to be plotted vs epochs
    :param location_plot: string that has location where plot will be saved
    :return: None
    """

    # add title to plot
    plt.title(plot_title)
    # if there is a value assigned to string_metric, add plot of such metric
    # vs. epoch, for both training and validation
    if string_metric != '':
        plt.plot(history.history[string_metric])
        plt.plot(history.history['val_'+string_metric])
    # Add plot of loss vs epoch for both training and validation
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_' + 'loss'], '')
    # add label to X axis
    plt.xlabel("Epochs")
    # if there is a value assigned to string_metric, add legend plot of such metric
    # vs. epoch, for both training and validation and similar for loss in both
    # datasets. Only add plot of loss otherwise
    if string_metric != '':
        plt.legend([string_metric, 'val_'+string_metric, 'loss', 'val_loss'])
    else:
        plt.legend(['loss', 'val_loss'])

    # add grid to plot
    plt.grid(True, which='major', linestyle='-')
    # set vertical range to [0, 1.02]
    plt.gca().set_ylim(0, 1.02)
    # save plot to file and display message for user
    plt.savefig(location_plot+'performance_'+plot_title+'.png')
    print('image saved to ', location_plot+'performance_'+plot_title+'.png')
    # clears plot to avoid overlap with future plots coming from future calls
    plt.clf()
    #plt.show()


def generate_regression_scatter_plot(predicted_values: np.ndarray,
                                     actual_values: np.ndarray,
                                     model_name: str,
                                     plot_location: str) -> None:

    line_start = actual_values.min()
    line_end = actual_values.max()
    plt.title(model_name)
    plt.scatter(x=actual_values, y=predicted_values)
    plt.plot([line_start, line_end],
             [line_start, line_end],
             'k-', color='r')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    # save the recently populated array of images to file
    plt.savefig(plot_location + 'predicted_scatter_plot_' + model_name + '.png')
    # clears plot to avoid overlap with future plots coming from future calls
    plt.clf()
    #plt.show()


def evaluate_regression_model(model_: tf.keras.models.Sequential,
                              model_name: str,
                              train_set: List[np.ndarray],
                              test_set: List[np.ndarray],
                              num_epochs: int,
                              location_plot: str) -> None:

    """
    Function that evaluates a regression model; generates plot to show performance
    graphically
    :param model_: TF2 sequential model
    :param model_name: string that labels name of given model
    :param train_set: numpy array that captures train set
    :param test_set: numpy array that captures test set
    :param num_epochs: integer that captures number of epochs
    :param location_plot: string that indicates where plot will be saved to
    :return: None
    """

    # compile and fit model....
    model_.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-3))

    history = model_.fit(train_set[0], train_set[1],
                         validation_data=(test_set[0], test_set[1]),
                         epochs=num_epochs)

    # then evaluate on test set
    model_.evaluate(test_set[0], test_set[1], verbose=2)

    # generate plots associated with models performance
    ##############
    ##############
    plot_graphs(history, plot_title=model_name,
                location_plot=location_plot, string_metric='')

    # generate scatter plot

    y_predicted = model_.predict(test_set[0])
    y_actual = test_set[1]

    ############
    ############

    generate_regression_scatter_plot(predicted_values=y_predicted,
                                     actual_values=y_actual,
                                     model_name=model_name,
                                     plot_location=location_plot)


