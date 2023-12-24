"""
Module containing activation functions

Author: Pierpasquale Colagrande
Year: 2023
"""

import numpy as np


# define a function implementing the sign activation function
def sign_activation(x: np.array) -> np.array:
    """
    Function implementing the sign activation function, turning values greater than 0 to and values lower than 0 to -1

    :param x: the numpy array over which to apply the activation function (np.array)
    :return: the array with the activation function applied on it (np.array)
    """

    return np.where(x >= 0, 1, -1)
