"""
Module containing visualization methods to visualize classification results on graph

Author: Pierpasquale Colagrande
Year: 2023
"""

import numpy as np
from matplotlib.axes import Axes


def plot_result(plot: Axes, x: np.array, y_true: np.array, y_pred: np.array, title: str) -> Axes:
    """
    Function that plots data accordingly to the classification results. Correctly classified samples are plotted in
    green, while misclassified samples are plotted in red. The marker is 'o' for class 1 and 'x' for class -1.

    :param plot: the pyplot axes over which to plot the data (matplotlib.axes.Axes)
    :param x: the input data over which classification was done; shape is (B, F), where B is the batch size and F is the
              number of features, which should be equal to the number of weights of the perceptron (np.array)
    :param y_true: the batch of ground truth labels; shape is (B), where B is the batch size (np.array)
    :param y_pred: the batch of predicted labels; shape is (B), where B is the batch size (np.array)
    :param title: the title of the plot (str)
    :return: the updated axes with the plotted data (matplotlib.axes.Axes)
    """

    # separate points based on their class and correctness
    correctly_classified = x[y_true == y_pred]
    y_correctly_classified = y_pred[y_true == y_pred]
    misclassified = x[y_true != y_pred]
    y_misclassified = y_pred[y_true != y_pred]

    # plot correctly classified points in green, with different marker styles accordingly to the classification class
    plot.scatter(correctly_classified[y_correctly_classified == 1][:, 0],
                 correctly_classified[y_correctly_classified == 1][:, 1],
                 marker='o', c='green', label='Class 1 (Correct)')
    plot.scatter(correctly_classified[y_correctly_classified == -1][:, 0],
                 correctly_classified[y_correctly_classified == -1][:, 1],
                 marker='x', c='green', label='Class -1 (Correct)')

    # plot misclassified points in red, with different marker styles accordingly to the classification class
    plot.scatter(misclassified[y_misclassified == 1][:, 0], misclassified[y_misclassified == 1][:, 1],
                 marker='o', c='red', label='Class 1 (Misclassified)')
    plot.scatter(misclassified[y_misclassified == -1][:, 0], misclassified[y_misclassified == -1][:, 1],
                 marker='x', c='red', label='Class -1 (Misclassified)')

    # add labels and legend
    plot.set_title(title)
    plot.set_xlabel('Feature 1')
    plot.set_ylabel('Feature 2')
    plot.legend(loc='lower left', fontsize="small")

    return plot
