"""
Perceptron module containing the class defining the perceptron

Author: Pierpasquale Colagrande
Year: 2023
"""

import numpy as np
from typing import Callable, Tuple
from src.activations import sign_activation
from src.metrics import accuracy


# define a perceptron class defining the perceptron structure
class Perceptron:
    """
    Class defining a Perceptron
    """

    def __init__(self, n_weights: int = 10, bias: bool = True, weight_init: str = "std",
                 activation_function: Callable = sign_activation) -> None:
        """
        The constructor method of the Perceptron that initializes a perceptron with a given number of weights.

        :param n_weights: number of weights of the perceptron, corresponding to the number of features
                          of the data (int, default 10)
        :param bias: boolean defining if the perceptron has a bias or not (bool, default True)
        :param weight_init: string defining how to initialize the weights; possible values are:
                            - "zero": set initial weights to 0
                            - "std": set initial weights to random numbers between 0 an 1
                            - "step": set initial weights to random numbers between -1 and 1
                            (str, default "std")
        :param activation_function: the activation function of the perceptor
                                    (Callable, default "src.activations.sign_activation")
        :return: None
        """

        # set the activation function
        self.activation_function = activation_function

        # if the bias is set to true
        self.bias = bias
        if self.bias:
            # increment the number of weights to add the bias
            n_weights += 1

        # initialize the weights (+ bias) as an array of n_weights number, depending on the specified method
        # if the bias is required, we will consider the last element of the array as the bias
        if weight_init == "step":
            self.weights = np.random.uniform(low=-1, high=1, size=(n_weights,))

        elif weight_init == "zero":
            self.weights = np.zeros(shape=(n_weights,))
        else:
            self.weights = np.random.rand(n_weights)

    def train(self, x: np.array, y: np.array, num_epochs: int = 1000, learning_rate: float = 0.01,
              verbose: bool = False) -> Tuple[np.array, float]:
        """
        Method performing the training of the perceptron accordingly to the Perceptron Learning Algorithm. The training
        ends when the max number of epoch is reached or all samples are correctly classified

        :param x: the input data to use for the training; shape is (B, F), where B is the batch size and F is the number
                  of features, which should be equal to the number of weights of the perceptron (np.array)
        :param y: the ground truth of the classes (labels) of the input samples; shape is (B), where B is the batch size
                  (np.array)
        :param num_epochs: the number of training epochs (int, default 1000)
        :param learning_rate: the learning rate of the perceptron (float, default 0.01)
        :param verbose: a boolean setting the verbosity of the training (bool, default True)
        :return: a tuple containing the predicted labels at the last epoch and the training accuracy
                 (Tuple[np.array, float)
        """

        # initialize the error rate to the number of samples in the set
        error_rate = x.shape[0]

        # initialize y_pred to true values of y
        y_pred = y

        # initialize epoch to first epoch (epoch 0)
        epoch = 0

        # iterate for the specified number of epochs
        while error_rate != 0 and epoch < num_epochs:
            # perform a forward pass feeding the input data to the perceptor
            y_pred = self.__forward(x)

            # compute the error rate for the current classification
            errors = y - y_pred

            # compute the update for the weights
            update = learning_rate * errors

            # if bias is being used, add a columns of 1 to the input data to also update the bias
            if self.bias:
                x_for_update = np.c_[x, np.ones(x.shape[0])]
            else:
                x_for_update = x

            # update the weights
            update = np.dot(update, x_for_update)
            self.weights += update

            # compute the error rate
            error_rate = np.sum(y != y_pred)

            # if in verbose mode
            if verbose:
                print(f"Epoch {epoch + 1}: y_pred = {y_pred}, y_true = {y}, error_rate = {error_rate}")

            # increment the current number of epochs
            epoch += 1

        return y_pred, accuracy(y_true=y, y_pred=y_pred)

    def __forward(self, x: np.array) -> np.array:
        """
        Function performing a forward pass of the perceptron with a batch of samples

        :param x: the batch of features; shape is (B, F), where B is the batch size, or number of samples, and F is the
                  number of features, that should be equal to the number of weights of the perceptor (np.array)
        :return: a numpy array with the result of the forward pass, a.k.a. the predicted labels (np.array)
        """

        # extract only the weights if there's also a bias term
        if self.bias:
            weights = self.weights[:-1]
        else:
            weights = self.weights

        # compute the dot product between weights and input samples features (the weighted sum between weights and input
        # features)
        weighted_sum = np.dot(x, weights)

        # add the bias term if specified
        if self.bias:
            weighted_sum += self.weights[-1]

        # compute the activation of the weighted sum to get the output label predictions
        y = self.activation_function(weighted_sum)

        return y

    def test(self, x: np.array, y: np.array) -> Tuple[np.array, float]:
        """
        Method performing the prediction labels corresponding to given input data. It is simply calls the forward method
        on the given data.

        :param x: the batch of features; shape is (B, F), where B is the batch size, or number of samples, and F is the
                  number of features, that should be equal to the number of weights of the perceptor (np.array)
        :param y: the ground truth of the classes (labels) of the input samples; shape is (B), where B is the batch size
                  (np.array)
        :return: a tuple containing a numpy array with the result of the prediction, a.k.a. the predicted labels, and
                 the testing accuracy (Tuple[np.array, float])
        """

        # compute the predictions
        y_pred = self.__forward(x)

        return y_pred, accuracy(y_true=y, y_pred=y_pred)
