import scipy.optimize
from climin import *
import itertools
# from sklearn.utils import shuffle
from lossDeriv import *


class LBFGS(object):

    """docstring for LBFGS"""

    def __init__(self, ae, evaluate, theta, lossDeriv, train, test,
                 nn, modelArgs, iterCounter, batch_size, max_iter):
        super(LBFGS, self).__init__()
        self.ae = ae
        self.evaluate = evaluate
        self.theta = theta
        self.lossDeriv = lossDeriv
        self.train = train
        self.test = test
        self.nn = nn
        self.modelArgs = modelArgs
        self.iterCounter = iterCounter
        self.batch_size = batch_size
        self.max_iter = max_iter

    def __iter__(self):
        return self

    def next(self):
        outLayer = self.ae.nn.layers[-1]

        def cbk(x):
            if (self.iterCounter.count % 5) == 0:
                self.ae.setParameters(x)
                if outLayer.isPartial():
                    rmse, mae = self.evaluate.calculateRMSEandMAE(
                        self.train, self.test)
                else:
                    rmse, mae = self.evaluate.calculateRMSEandMAE(
                        self.test, self.test)
                print 'Iteration : %d '\
                    'Test RMSE: %f MAE: %f' % (
                        self.iterCounter.count, rmse, mae)

        opt_solution = scipy.optimize.minimize(self.lossDeriv,
                                               self.theta,
                                               args=(
                                                   self.train,
                                                   self.ae.nn, self.modelArgs,
                                                   self.iterCounter,
                                                   self.batch_size),
                                               method = 'L-BFGS-B',
                                               jac = True, callback=cbk,
                                               options =
                                               {'maxiter': self.max_iter,
                                                "disp": 0})

        opt_theta = opt_solution.x
        self.ae.setParameters(opt_theta)
        raise StopIteration("End of the iteration")


def getMiniBatchParamsIterator(train, nn, modelArgs, iterCounter,
                               batch_size, fn):
    m, n = train.shape
    batches = range(0, m, batch_size)
    if batches[-1] != m:
        batches.append(m)
    while True:
        # train = shuffle(train)
        for i in range(len(batches) - 1):
            start = batches[i]
            end = batches[i + 1]
            batch_data = train[start:end, :]
            yield ([batch_data, nn, modelArgs, iterCounter, batch_size,
                    fn], {})


def fprime(theta, user_item_rating, NN, modelArg, counter, batch_size, fn):
    cost, deriv = fn(
        theta, user_item_rating, NN, modelArg, counter, batch_size)
    return deriv


def getOptimizer(optimize, ae, evaluate, theta, train, test,
                 nn, modelArgs, iterCounter, batch_size, max_iter):

    if optimize == "lbfgs":
        optimizer = LBFGS(ae, evaluate, theta, getCostDeriv, train, test,
                          nn, modelArgs, iterCounter, batch_size, max_iter)
    elif optimize == "rprop":
        args = itertools.repeat(
            ([train, ae.nn, modelArgs, iterCounter, batch_size, getCostDeriv],
                {}))
        optimizer = rprop.Rprop(theta, fprime, args=args)
    elif optimize == "rmsprop":
        args = getMiniBatchParamsIterator(
            train, ae.nn, modelArgs, iterCounter, batch_size,
            getCostDerivBatch)
        optimizer = rmsprop.RmsProp(
            theta, fprime, 0.001, decay=0.0, step_adapt=False, step_rate_min=0,
            step_rate_max=5.0, args=args)
    else:
        raise NotImplementedError("%s optimizer not implemented" % optimize)
    return optimizer
