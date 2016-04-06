import numpy as np
from cython_activations import *


class Activation(object):

    """docstring for Activation"""

    def activation(self, x):
        pass

    def derivative(self, x):
        pass

    def binarize(self, x):
        return x


class Identity(Activation):

    """docstring for Identity"""

    def activation(self, x):
        return x

    def derivative(self, x):
        return 1


class Sigmoid(Activation):

    """docstring for Sigmoid"""

    def activation(self, x):
        if len(x.shape) == 2:
            return cy_sigmoid(x)
        else:
            return cy_sigmoid1d(x)

    def derivative(self, x):
        return np.multiply(x, 1 - x)

    def binarize(self, x):
        return 1.0 * (x > 0.5)


class RELU(Activation):

    """docstring for RELU"""

    def activation(self, x):
        return x * (x > 0)

    def derivative(self, x):
        return (x > 0) * 1


class NRELU(Activation):

    """docstring for NRELU"""

    def activation(self, x):
        if len(x.shape) == 2:
            sigma = cy_sigmoid(x)
        else:
            sigma = cy_sigmoid1d(x)
        x += np.random.randn(x.shape[0], x.shape[1]) * np.sqrt(sigma)
        return x * (x > 0)

    def derivative(self, x):
        return (x > 0) * 1


class Tanh(Activation):

    """docstring for RELU"""

    def activation(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return (1 - np.power(x, 2))

    def binarize(self, x):
        return 1.0 * (x > 0)
