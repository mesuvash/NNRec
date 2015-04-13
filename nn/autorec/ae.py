import numpy as np
import cPickle as pkl
from cython_matmul import *
from lossDeriv import *
from nn.blocks.activations import *
from nn.blocks.nn import *


class AE:

    def __init__(self, nn, modelArgs, debug=True):
        self.nn = nn
        self.debug = debug
        self.modelArgs = modelArgs
        self.nn.setLimits()

    def setParameters(self, theta):
        weights = []
        for i in range(len(self.nn.weights_limit) - 1):
            weight = theta[self.nn.weights_limit[i]:self.nn.weights_limit[
                i + 1]].reshape(self.nn.weights[i].shape)
            weights.append(weight)

        biases = []
        offset = self.nn.weights_limit[-1]
        for i in range(len(self.nn.bias_limit) - 1):
            bias = theta[offset + self.nn.bias_limit[i]:offset +
                         self.nn.bias_limit[i + 1]]
            bias = bias.reshape(self.nn.layers[i + 1].bias.shape)
            biases.append(bias)
        self.nn.weights = weights
        self.nn.biases = biases

    def getParameters(self):
        params = []
        for weight in self.nn.weights:
            params.append(weight.flatten())
        for bias in self.nn.biases:
            params.append(bias.flatten())
        return np.concatenate(params)

    def predict(self, train, test):

        inputActivation = train
        for i in range(len(self.nn.layers) - 2):
            if scipy.sparse.isspmatrix(inputActivation):
                forward = inputActivation * self.nn.weights[i]
            else:
                forward = np.dot(inputActivation, self.nn.weights[i])
            if self.nn.layers[i].dropout is not None:
                forward *= (1 - self.nn.layers[i].dropout)
            inputActivation = self.nn.layers[
                i + 1].activation.activation(forward + self.nn.biases[i])
            if self.nn.layers[i + 1].isBinary():
                inputActivation = self.nn.layers[
                    i + 1].activation.binarize(inputActivation)
        output_layer = self.nn.layers[-1]
        if output_layer.isPartial():
            output = multiplyOuterSparseLayer(inputActivation,
                                              self.nn.weights[-1],
                                              self.nn.biases[-1], test.data,
                                              test.indices, test.indptr,
                                              self.modelArgs.num_threads)
        else:
            output = np.dot(
                inputActivation, self.nn.weights[-1]) + self.nn.biases[-1]

        if self.nn.layers[-2].hasDropout():
            output *= (1 - self.nn.layers[-2].dropout)

        output = output_layer.activation.activation(output)
        if self.modelArgs.mean > 0.0:
            output += self.modelArgs.mean

        if output_layer.isPartial():
            _max, _min = train.data.max(), train.data.min()
            output[output > _max] = _max
            output[output < _min] = _min
            output = scipy.sparse.csr_matrix((output, test.indices,
                                              test.indptr), shape=test.shape)

        return output

    def getActivationOfLayer(self, train, layerno):
        inputActivation = train
        assert((layerno > 0) and (layerno < len(self.nn.layers)))
        for i in range(layerno):
            if scipy.sparse.isspmatrix(inputActivation):
                forward = inputActivation * self.nn.weights[i]
            else:
                forward = np.dot(inputActivation, self.nn.weights[i])
            if self.nn.layers[i].dropout is not None:
                forward *= (1 - self.nn.layers[i].dropout)
            inputActivation = self.nn.layers[
                i + 1].activation.activation(forward + self.nn.biases[i])
            if self.nn.layers[i + 1].isBinary():
                inputActivation = self.nn.layers[
                    i + 1].activation.binarize(inputActivation)
        return inputActivation

    def saveModel(self, path):
        print "Saving model to path : ", path
        pkl.dump(self, open(path, "wb"))
