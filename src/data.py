"""
Module containing the functions to manage the dataset

Author: Pierpasquale Colagrande
Year: 2023
"""

import numpy as np
from typing import Tuple


# define a function that reads a dataset encoded in a txt file and returns it into a numpy array
def read_dataset(file_path: str) -> Tuple[np.array, int, int]:
    """
    Reads a dataset encoded as .txt file in the following format
    num_samples num_features
    x[0, 0] x[0, 1] ... x[0, num_features - 1] y[0]
    x[1, 0] x[1, 1] ... x[1, num_features - 1] y[1]
    ...
    x[num_samples-1, 0] x[num_samples-1, 1] ... x[num_samples-1, num_features - 1] y[num_samples - 1]

    :param file_path: the path of the .txt file to read (str)
    :return: a tuple containing the dataset as a numpy array, the number of samples of the dataset and the number of
             features of the dataset (Tuple[np.array, int, int])
    """

    # open the file in read mode and extract the data
    with open(file_path, 'r') as file:
        # read the first line to get the number of samples and features
        num_samples, num_features = map(int, file.readline().split())

        # read the remaining lines and extract the data
        data = np.array([list(map(int, line.split())) for line in file])

    # return the tuple
    return data, num_samples, num_features
