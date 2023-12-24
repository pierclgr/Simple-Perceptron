from src.data import read_dataset
from src.perceptron import Perceptron
from src.activations import sign_activation
from src.visualization import plot_result
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main method of the program. It uses the hydra configuration file config.yaml in the config directory.

    :param cfg: the hydra dictionary configuration taken from the configuration file (omegaconf.DictConf)
    :return: None
    """

    # extract parameters from the configuration
    train_file_path = cfg.train_set_path
    test_file_path = cfg.test_set_path
    bias = cfg.bias
    weight_init = cfg.weight_init
    activation = cfg.activation
    num_epochs = cfg.num_epochs
    learning_rate = cfg.learning_rate
    verbose = cfg.verbose

    # set activation function
    if activation == "sign_activation":
        activation = sign_activation

    # read the training set
    train_set, n_train_samples, n_train_features = read_dataset(file_path=train_file_path)

    # read the testing set
    test_set, n_test_samples, n_test_features = read_dataset(file_path=test_file_path)

    # split the training set into features and labels
    x_train = train_set[:, :-1]
    y_train = train_set[:, -1]

    # split the testing set into features and labels
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]

    # dataset validation
    # raise an exception if the number of features in the training set is different from the one specified in the
    # training set file
    train_set_features = x_train.shape[1]
    if n_train_features is not train_set_features:
        raise Exception(
            f"The number of features specified in the training set file is different from the actual number of features"
            f" ({n_train_features}, {train_set_features}).")

    # raise an exception if the number of features in the test set is different from the one specified in the test set
    # file
    test_set_features = x_test.shape[1]
    if n_test_features is not test_set_features:
        raise Exception(
            f"The number of features specified in the test set file is different from the actual number of features"
            f" ({n_test_features}, {test_set_features}).")

    # raise an exception if the number of features in the test set is different from the number of features in the
    # training set
    if train_set_features is not test_set_features:
        raise Exception(
            f"Training and test set have different number of features ({train_set_features}, {test_set_features}).")

    # initialize the perceptron
    perceptron = Perceptron(n_weights=train_set_features, bias=bias, weight_init=weight_init,
                            activation_function=activation)

    # train the perceptron on the training set
    y_pred_train, training_accuracy = perceptron.train(x_train, y_train, num_epochs=num_epochs,
                                                       learning_rate=learning_rate,
                                                       verbose=verbose)

    # compute the predictions with the trained perceptron on the test set
    y_pred_test, test_accuracy = perceptron.test(x_test, y_test)

    # print the training and testing accuracy
    print(f"Training accuracy: {training_accuracy} %")
    print(f"Test accuracy: {test_accuracy} %")

    # initialise the subplot function using 2 rows
    _, axis = plt.subplots(2)

    # plot the result of training and testing classification
    axis[0] = plot_result(plot=axis[0], x=x_train, y_true=y_train, y_pred=y_pred_train,
                          title=f"Training, accuracy {training_accuracy} %")
    axis[1] = plot_result(plot=axis[1], x=x_test, y_true=y_test, y_pred=y_pred_test,
                          title=f"Test, accuracy {test_accuracy} %")

    # show the plot
    plt.show()


if __name__ == '__main__':
    main()
