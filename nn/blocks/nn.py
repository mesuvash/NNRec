import numpy as np
from copy import deepcopy


class LayerType(object):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class Layer:

    def __init__(self, num_units, activation, layerType,
                 dropout=None, sparsity=None, partial=False, isBiasEnabled=True):
        self.num_units = num_units
        self.activation = activation
        self.mf = True
        self.dropout = dropout
        self.layerType = layerType
        self.sparsity = sparsity
        self.partial = partial
        self.isBiasEnabled = True
        self.binary = False
        self.setBias()

    def setBias(self):
        self.bias = np.random.randn(1, self.num_units) * 0.001

    def setSparsity(self, value):
        self.sparsity = value

    def isSparse(self):
        return ((self.sparsity is not None) and (self.sparsity != 1))

    def setPartial(self):
        self.partial = True

    def isPartial(self):
        return self.partial

    def setBinary(self):
        self.binary = True

    def isBinary(self):
        return (self.binary == True)

    def setDropout(self, p):
        self.dropout = p

    def hasDropout(self):
        return ((self.dropout is not None) and (self.dropout != 0.0))

    def hasBias(self):
        return (hasattr(self, "bias") and (self.bias is not None))

    def removeBias(self):
        self.isBiasEnabled = False
        self.bias = np.zeros((1, self.num_units))

    def unsetMeanField(self):
        self.mf = False

    def copy(self):
        return deepcopy(self)

    def __str__(self):

        layerinfo = "Number of Units = %d ; Layer type = %s\n" % (self.num_units,
                                                                  self.activation)

        drp = self.dropout if self.dropout else 0
        sps = self.sparsity if self.sparsity else 0

        additional_info = "Sparsity %f \t Dropout %f \t Partial %r " % (
            sps, drp, self.partial)
        return layerinfo + additional_info


class NN(object):

    def __init__(self):
        self.layers = []
        self.weights = []

    def _add_weights(self, n1, n2):
        w_vis2hid = 0.01 * np.random.randn(n1, n2)
        self.weights.append(w_vis2hid)

    def addLayer(self, layer1):
        self.layers.append(layer1)
        if (len(self.layers) > 1):
            self._add_weights(
                self.layers[-2].num_units, self.layers[-1].num_units)

    def getWeightByIndex(self, index):
        return self.weights[index]

    def setLimits(self):
        self.weights_limit = [0]
        self.bias_limit = [0]
        for l in range(len(self.layers) - 1):
            self.weights_limit.append(
                self.weights_limit[-1] + self.layers[l].num_units *
                self.layers[l + 1].num_units)

            self.bias_limit.append(
                self.bias_limit[-1] + self.layers[l + 1].num_units)

    def getFlattenParams(self):
        params = []
        map(lambda x: params.append(x.flatten()), self.weights)
        map(lambda x: params.append(x.bias.flatten()), self.layers[1:])
        return np.concatenate(params)

    def finalize(self):
        self.setLimits()

    def setDropout(self, layerIndex, dropout_prob):
        self.layers[layerIndex].setDropout(dropout_prob)
        self.weights[layerIndex] *= (1 / (1 - dropout_prob))

    def __str__(self):
        representation = ""
        for i, l in enumerate(self.layers):
            representation += "Layer = %d ; " % i + str(l) + "\n"
        return representation
