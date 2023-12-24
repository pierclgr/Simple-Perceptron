"""
Module containing some classification metrics

Author: Pierpasquale Colagrande
Year: 2023
"""

import numpy as np


# define a method to compute the accuracy
def accuracy(y_true: np.array, y_pred: np.array) -> float:
    """
    Function computing the accuracy of a prediction between ground truth labels and predicted labels

    :param y_true: the batch of ground truth labels; shape is (B), where B is the batch size (np.array)
    :param y_pred: the batch of predicted labels; shape is (B), where B is the batch size (np.array)
    :return: the accuracy of the prediction in the 0-100 range (float)
    """

    return 100 * (np.sum(y_true == y_pred) / y_true.shape[0])
