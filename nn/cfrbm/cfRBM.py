import numpy as np
from cython_rbm_matmul import cython_binarizeSparseMatrix, multiplyOuterSparseLayer
from utils.metrics.evaluate import EvaluateRBM


class ModelArgs(object):

    """docstring for ModelArgs"""

    def __init__(self, learn_rate=0.001, regularize_bias=True,
                 momentum=0.6, lamda=0.001, CD=1, num_threads=10, max_iter=200,
                 k=5, mapping=None, min_learn_rate=10e-6,
                 batch_size=None):
        super(ModelArgs, self).__init__()
        self.learn_rate = learn_rate
        self.regularize_bias = regularize_bias
        self.CD = CD
        self.momentum = momentum
        self.lamda = lamda
        self.num_threads = num_threads
        self.max_iter = max_iter
        self.k = 5
        self.mapping = mapping
        self.min_learn_rate = min_learn_rate
        self.batch_size = batch_size

    def __str__(self):
        string = ""
        for key in self.__dict__.keys():
            string += "%s: %s\t" % (key, str(self.__dict__[key]))
        return string


def binarizeSparseMatrix(x, k, mapping):
    m, n = x.shape
    return cython_binarizeSparseMatrix(x.data, x.indices, x.indptr,
                                       m, n, k, mapping)


class RBM(object):

    """docstring for RBM"""

    def __init__(self, nn, modelArgs, debug=True):
        super(RBM, self).__init__()
        self.nn = nn
        self.modelArgs = modelArgs
        self.debug = debug
        ratings_array = modelArgs.mapping.keys()
        ratings_array.sort()
        ratings_array = np.array(ratings_array)
        self.ratings_array = ratings_array.reshape((modelArgs.k, 1))

    def getHiddenActivation(self, x):
        hidden = x * self.nn.weights[0] + self.nn.layers[1].bias
        hidden = self.nn.layers[1].activation.activation(hidden)
        if self.nn.layers[1].isBinary():
            hidden = self.nn.layers[1].activation.binarize(hidden)
        return hidden

    def getVisibleActivation(self, x, target, ncpus=16):
        visible = multiplyOuterSparseLayer(x, self.nn.weights[0].T,
                                           self.nn.layers[0].bias,
                                           target.data,
                                           target.indices,
                                           target.indptr,
                                           ncpus)
        return self.nn.layers[0].activation.activation(visible)

    def __binary2Ratings(self, prediction):
        n = self.modelArgs.k
        m = int(len(prediction) / n)
        prediction = prediction.reshape(m, n)
        normalizer = prediction.sum(axis=1).reshape(m, 1)
        prediction = prediction / normalizer
        rating = np.dot(prediction, self.ratings_array)
        return np.ravel(rating)

    def predict(self, train, test, normalize=True):
        hidden = self.getHiddenActivation(train)
        visible = self.getVisibleActivation(hidden, test)
        # visible = np.exp(visible)
        if normalize:
            prediction = self.__binary2Ratings(visible)
        else:
            prediction = visible
        return prediction


class RbmOptimizer(object):

    """docstring for RbmOptimizer"""

    def __init__(self, RBM):
        super(RbmOptimizer, self).__init__()
        self.RBM = RBM

    def train(self, train, test, rtest):
        self.nn = self.RBM.nn
        learn_rate = self.RBM.modelArgs.learn_rate
        max_iter = self.RBM.modelArgs.max_iter
        CD = self.RBM.modelArgs.CD
        lamda = self.RBM.modelArgs.lamda
        momentum = self.RBM.modelArgs.momentum
        min_learn_rate = self.RBM.modelArgs.min_learn_rate

        dW_old = np.zeros(self.nn.weights[0].shape)
        dv_old = np.zeros(self.nn.layers[0].bias.shape)
        dh_old = np.zeros(self.nn.layers[1].bias.shape)
        evaluate = EvaluateRBM(self.RBM)

        vispos = train
        visneg = train.copy()
        for i in range(max_iter):
            if i > 50:
                CD = 3
                momentum = 0.9

            hidpos = self.RBM.getHiddenActivation(vispos)
            hidneg = hidpos
            for j in range(CD):
                visneg_data = self.RBM.getVisibleActivation(hidneg, vispos)
                visneg.data = visneg_data
                hidneg = self.RBM.getHiddenActivation(visneg)

            dW = momentum * dW_old + learn_rate *\
                ((vispos.T * hidpos) -
                 (visneg.T * hidneg) - lamda * self.nn.weights[0])
            dvbias = momentum * dv_old + 0.1 * learn_rate *\
                ((vispos - visneg).sum(axis=0) -
                 lamda * self.nn.layers[0].bias)
            dhbias = momentum * dh_old + learn_rate *\
                ((hidpos - hidneg).sum(axis=0) -
                 lamda * self.nn.layers[1].bias)

            dW_old = dW
            dv_old = dvbias
            dh_old = dhbias

            self.nn.weights[0] += dW
            self.nn.layers[0].bias += dvbias
            self.nn.layers[1].bias += dhbias
            if i % 5 == 0:
                learn_rate = max(learn_rate * 0.95, min_learn_rate)
                print evaluate.calculateRMSEandMAE(train, test, rtest)

    def minibatchTrain(self, train, test, rtest, batch_size):
        self.nn = self.RBM.nn
        slearn_rate = self.RBM.modelArgs.learn_rate
        max_iter = self.RBM.modelArgs.max_iter
        CD = self.RBM.modelArgs.CD
        lamda = self.RBM.modelArgs.lamda
        momentum = self.RBM.modelArgs.momentum
        min_learn_rate = self.RBM.modelArgs.min_learn_rate

        dW_old = np.zeros(self.nn.weights[0].shape)
        dv_old = np.zeros(self.nn.layers[0].bias.shape)
        dh_old = np.zeros(self.nn.layers[1].bias.shape)
        evaluate = EvaluateRBM(self.RBM)


        m, n = train.shape
        batches = range(0, m, batch_size)
        if batches[-1] != m:
            if (m - batches[-1]) < (batch_size / 2.0):
                batches[-1] = m
            else:
                batches.append(m)
        for i in range(max_iter):
            if i > 50:
                CD = 3
                momentum = 0.9
            for j in range(len(batches) - 1):
                start = batches[j]
                end = batches[j + 1]
                learn_rate = slearn_rate / (end - start)
                learn_rate = max(learn_rate, min_learn_rate)

                vispos = train[start:end, :]
                visneg = vispos.copy()
                hidpos = self.RBM.getHiddenActivation(vispos)
                hidneg = hidpos
                for k in range(CD):
                    visneg_data = self.RBM.getVisibleActivation(hidneg, vispos)
                    visneg.data = visneg_data
                    hidneg = self.RBM.getHiddenActivation(visneg)

                dW = momentum * dW_old + learn_rate *\
                    ((vispos.T * hidpos) -
                     (visneg.T * hidneg) - lamda * self.nn.weights[0])
                dvbias = momentum * dv_old + learn_rate *\
                    ((vispos - visneg).sum(axis=0) -
                     lamda * self.nn.layers[0].bias)
                dhbias = momentum * dh_old + 0.1 * learn_rate *\
                    ((hidpos - hidneg).sum(axis=0) -
                     lamda * self.nn.layers[1].bias)

                dW_old = dW
                dv_old = dvbias
                dh_old = dhbias

                self.nn.weights[0] += dW
                self.nn.layers[0].bias += dvbias
                self.nn.layers[1].bias += dhbias
            if i % 5 == 0:
                slearn_rate *= 0.95
                print evaluate.calculateRMSEandMAE(train, test, rtest)

    def sgdTrain(self, train, test, rtest):
        self.nn = self.RBM.nn
        learn_rate = self.RBM.modelArgs.learn_rate
        max_iter = self.RBM.modelArgs.max_iter
        CD = self.RBM.modelArgs.CD
        lamda = self.RBM.modelArgs.lamda
        momentum = self.RBM.modelArgs.momentum

        dW_old = np.zeros(self.nn.weights[0].shape)
        dv_old = np.zeros(self.nn.layers[0].bias.shape)
        dh_old = np.zeros(self.nn.layers[1].bias.shape)
        evaluate = EvaluateRBM(self.RBM)
        # traindata = train.data
        # testdata = test.data

        m, n = train.shape
        for i in range(max_iter):
            if i > 50:
                CD = 3
                momentum = 0.9
            for j in range(m - 1):
                vispos = train.getrow(j)
                visneg = vispos.copy()
                hidpos = self.RBM.getHiddenActivation(vispos)
                hidneg = hidpos
                for k in range(CD):
                    visneg_data = self.RBM.getVisibleActivation(hidneg, vispos)
                    visneg.data = visneg_data
                    hidneg = self.RBM.getHiddenActivation(visneg)

                dW = momentum * dW_old + learn_rate *\
                    ((vispos.T * hidpos) -
                     (visneg.T * hidneg) - lamda * self.nn.weights[0])
                dvbias = momentum * dv_old + learn_rate *\
                    ((vispos - visneg).sum(axis=0) -
                     lamda * self.nn.layers[0].bias)
                dhbias = momentum * dh_old + 0.1 * learn_rate *\
                    ((hidpos - hidneg).sum(axis=0) -
                     lamda * self.nn.layers[1].bias)

                dW_old = dW
                dv_old = dvbias
                dh_old = dhbias

                self.nn.weights[0] += dW
                self.nn.layers[0].bias += dvbias
                self.nn.layers[1].bias += dhbias
            if i % 5 == 0:
                slearn_rate *= 0.95
                print evaluate.calculateRMSEandMAE(train, test, rtest)
