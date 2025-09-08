import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """Implements MultiLayerPerceptron (MLP) with single hidden layer.

    :param embedding_size: dimensionality of input layer (int)
    :param hidden_size: dimensionality of hidden layer (int)
    :param n_classes: dimensionality of output layer (int)
    """

    def __init__(self, embedding_size, hidden_size, n_classes):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, x):
        """Performs forward operation of model.

        :param x: input to MLP
        :return: logits
        """
        logits = self.layers(x)

        return logits

    def reset(self):
        """Re-initialises weights of model.
        """
        for layer in [0, 2, 4]:
            self.layers[layer].reset_parameters()


class LinearClassifier(nn.Module):
    """Implements linear classifier.

    :param embedding_size: dimensionality of input layer (int)
    :param n_classes: dimensionality of output layer (int)
    """

    def __init__(self, embedding_size, n_classes):
        super().__init__()

        self.linear = nn.Linear(embedding_size, n_classes)

    def forward(self, x):
        """Performs forward operation of model.

        :param x: input to MLP
        :return: logits
        """
        logits = self.linear(x)

        return logits

    def reset(self):
        """Re-initialises weights of model.
        """
        self.linear.reset_parameters()
