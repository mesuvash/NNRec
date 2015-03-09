import numpy as np
import scipy.sparse
from nn.blocks.cython_activations import *
from cython_matmul import *
from nn.blocks.nn import LayerType
# from sklearn.utils import shuffle
from copy import deepcopy


EPS = 10e-15


def _getLossUpdateDerivative(batch_data, weights, biases,
                             dWeights, dBiases, NN, modelArg):
    batch_shape = batch_data.shape
    ######################Forward pass######################
    fActivation = []
    layerInput = batch_data
    cost = 0.0
    for l, layer in enumerate(NN.layers):
        if layer.layerType == LayerType.INPUT:
            activation = layerInput
        elif layer.layerType == LayerType.HIDDEN:
            if scipy.sparse.isspmatrix(layerInput):
                x = layerInput * weights[l - 1] + biases[l - 1]
            else:
                x = np.dot(layerInput, weights[l - 1]) + biases[l - 1]
            activation = layer.activation.activation(x)
        elif layer.layerType == LayerType.OUTPUT:
            if layer.isPartial():
                x = multiplyOuterSparseLayer(layerInput, weights[l - 1],
                                             biases[l - 1],
                                             batch_data.data,
                                             batch_data.indices,
                                             batch_data.indptr,
                                             modelArg.num_threads)
                activation = layer.activation.activation(x)
                activation = scipy.sparse.csr_matrix((activation,
                                                      batch_data.indices,
                                                      batch_data.indptr),
                                                     shape=batch_shape)
            else:
                x = np.dot(layerInput, weights[l - 1]) + biases[l - 1]
                activation = layer.activation.activation(x)

        if (layer.dropout is not None) and (layer.dropout != 0):
            dropout(activation, layer.dropout)
        fActivation.append(activation)
        # binarize for the forward propagation
        if layer.isBinary():
            layerInput = layer.activation.binarize(activation)
        else:
            layerInput = activation

    ######################Calculate error######################
    # sparse csr matrix
    if NN.layers[-1].isPartial():
        diff = fActivation[-1].data - batch_data.data
    else:
        diff = fActivation[-1] - batch_data
    sum_of_squares_error = 0.5 * np.sum(np.power(diff, 2))
    cost += sum_of_squares_error

    ######################BackPropagation######################
    l = len(NN.layers) - 1
    for layer in NN.layers[::-1]:
        if layer.layerType == LayerType.OUTPUT:
            if layer.isPartial():
                delta = np.multiply(
                    diff, layer.activation.derivative(fActivation[l].data))
                delta = scipy.sparse.csr_matrix((delta,
                                                 batch_data.indices,
                                                 batch_data.indptr),
                                                shape=batch_shape)
            else:
                delta = np.multiply(
                    diff, layer.activation.derivative(fActivation[l]))

            if (scipy.sparse.isspmatrix(fActivation[l - 1]) or
                    scipy.sparse.isspmatrix(delta)):

                wderiv = fActivation[l - 1].T * delta
            else:
                wderiv = np.dot(fActivation[l - 1].T, delta)
            bderiv = delta.sum(axis=0)
            dWeights[l - 1] += wderiv
            dBiases[l - 1] += bderiv

        if layer.layerType == LayerType.HIDDEN:
            if layer.isSparse():
                rho_hat = fActivation[l].sum(
                    axis=0) / fActivation[l].shape[0]
                rho = layer.sparsity
                KL_divergence = modelArg.beta * np.sum(
                    rho * np.log(rho / rho_hat) +
                    (1 - rho) * np.log((1 - rho) / ((1 - rho_hat) + EPS)))
                cost += KL_divergence
                KL_grad = modelArg.beta * \
                    (-(rho / rho_hat) +
                     ((1 - rho) / ((1 - rho_hat) + EPS)))

            if scipy.sparse.issparse(delta):
                if layer.isSparse():
                    delta = np.multiply(
                        delta * weights[l].T + KL_grad,
                        layer.activation.derivative(fActivation[l]))
                else:
                    delta = np.multiply(
                        delta * weights[l].T,
                        layer.activation.derivative(fActivation[l]))
            else:
                if layer.isSparse():
                    delta = np.multiply(
                        np.dot(delta, weights[l].T) + KL_grad,
                        layer.activation.derivative(fActivation[l]))
                else:
                    delta = np.multiply(
                        np.dot(delta, weights[l].T),
                        layer.activation.derivative(fActivation[l]))

            if (scipy.sparse.isspmatrix(fActivation[l - 1])
                    or scipy.sparse.isspmatrix(delta)):
                wderiv = fActivation[l - 1].T * delta
            else:
                wderiv = np.dot(fActivation[l - 1].T, delta)
            dWeights[l - 1] += wderiv
            if layer.isBiasEnabled:
                bderiv = delta.sum(axis=0)
                dBiases[l - 1] += bderiv
        l = l - 1
    return cost


def getCostDeriv(theta, user_item_rating, NN,
                 modelArg, counter, batch_size):
    counter.increment()
    ##################################### Unrolling/ Initialization ##########
    weights = []
    for i in range(len(NN.weights_limit) - 1):
        weight = theta[NN.weights_limit[i]:NN.weights_limit[i + 1]]
        weight = weight.reshape(NN.weights[i].shape)
        weights.append(weight)

    biases = []
    offset = NN.weights_limit[-1]
    for i in range(len(NN.bias_limit) - 1):
        bias = theta[offset + NN.bias_limit[i]:offset +
                     NN.bias_limit[i + 1]].reshape(NN.layers[i + 1].bias.shape)
        biases.append(bias)

    dWeights = []
    for weight in weights:
        dWeights.append(np.zeros(shape=weight.shape))

    dBiases = []
    for bias in biases:
        dBiases.append(np.zeros(shape=bias.shape))

    ##################################### Batch loop #########################

    m, n = user_item_rating.shape
    batches = range(0, m, batch_size)
    if batches[-1] != m:
        batches.append(m)

    cost = 0.0
    for i in range(len(batches) - 1):
        start = batches[i]
        end = batches[i + 1]
        batch_data = user_item_rating[start:end, :]
        loss = _getLossUpdateDerivative(batch_data, weights, biases,
                                        dWeights, dBiases, NN, modelArg)
        cost += loss

    if not modelArg.regularize_bias:
        weight_decay = 0.5 *\
            reduce(
                lambda x, y: x + y, map(lambda z:
                                        np.power(
                                            weights[z], 2).sum() *
                                        modelArg.lamda[z],
                                        range(len(weights))))
    else:
        weight_decay = 0.5 *\
            reduce(
                lambda x, y: x + y, map(lambda z:
                                        np.power(
                                            weights[z], 2).sum() *
                                        modelArg.lamda[z],
                                        range(len(weights))))
        weight_decay = 0.5 *\
            reduce(
                lambda x, y: x + y, map(lambda z:
                                        np.power(
                                            biases[z], 2).sum() *
                                        modelArg.lamda[z],
                                        range(len(biases))))
    cost += weight_decay

    for i in range(len(dWeights)):
        # dWeights[i] += modelArg.lamda * weights[i]
        dWeights[i] += modelArg.lamda[i] * weights[i]

    if modelArg.regularize_bias:
        for i in range(len(dBiases)):
            # dBiases[i] += modelArg.lamda * biases[i]
            dBiases[i] += modelArg.lamda[i] * biases[i]

    theta_grad = np.concatenate(map(lambda x: x.flatten(), dWeights + dBiases))
    return [cost, theta_grad]


def getCostDerivBatch(theta, user_item_rating, NN,
                      modelArg, counter, batch_size):
    counter.increment()
    # user_item_rating = shuffle(user_item_rating)
    ##################################### Unrolling/ Initialization ##########
    weights = []
    for i in range(len(NN.weights_limit) - 1):
        weight = theta[NN.weights_limit[i]:NN.weights_limit[i + 1]]
        weight = weight.reshape(NN.weights[i].shape)
        weights.append(weight)

    biases = []
    offset = NN.weights_limit[-1]
    for i in range(len(NN.bias_limit) - 1):
        bias = theta[offset + NN.bias_limit[i]:offset +
                     NN.bias_limit[i + 1]].reshape(NN.layers[i + 1].bias.shape)
        biases.append(bias)

    dWeights = []
    for weight in weights:
        dWeights.append(np.zeros(shape=weight.shape))

    dBiases = []
    for bias in biases:
        dBiases.append(np.zeros(shape=bias.shape))

    ##################################### Batch loop #########################

    m, n = user_item_rating.shape
    batches = range(0, m, batch_size)
    if batches[-1] != m:
        batches.append(m)

    cost = 0.0
    for i in range(len(batches) - 1):
        start = batches[i]
        end = batches[i + 1]
        batch_data = user_item_rating[start:end, :]
        loss = _getLossUpdateDerivative(batch_data, weights, biases,
                                        dWeights, dBiases, NN, modelArg)
        cost += loss

        if not modelArg.regularize_bias:
            weight_decay = 0.5 *\
                reduce(
                    lambda x, y: x + y, map(lambda z:
                                            np.power(
                                                weights[z], 2).sum() *
                                            modelArg.lamda[z],
                                            range(len(weights))))
        else:
            weight_decay = 0.5 *\
                reduce(
                    lambda x, y: x + y, map(lambda z:
                                            np.power(
                                                weights[z], 2).sum() *
                                            modelArg.lamda[z],
                                            range(len(weights))))
            weight_decay = 0.5 *\
                reduce(
                    lambda x, y: x + y, map(lambda z:
                                            np.power(
                                                biases[z], 2).sum() *
                                            modelArg.lamda[z],
                                            range(len(biases))))
        cost += weight_decay

        for i in range(len(dWeights)):
            dWeights[i] += modelArg.lamda[i] * weights[i]

        if modelArg.regularize_bias:
            for i in range(len(dBiases)):
                dBiases[i] += modelArg.lamda[i] * biases[i]

        theta_grad = np.concatenate(
            map(lambda x: x.flatten(), dWeights + dBiases))
        return [cost, theta_grad]


def updateSGD(user_item_rating, NN, modelArg, counter, batch_size,
              alpha, dWeights_old, dBiases_old):
    counter.increment()

    # user_item_rating = shuffle(user_item_rating)
    weights = NN.weights
    biases = NN.biases

    dWeights = []
    for weight in weights:
        dWeights.append(np.zeros(shape=weight.shape))

    dBiases = []
    for bias in biases:
        dBiases.append(np.zeros(shape=bias.shape))

    ##################################### Batch loop #########################

    m, n = user_item_rating.shape
    batches = range(0, m, batch_size)
    if batches[-1] != m:
        batches.append(m)
    cost = 0.0
    for i in range(len(batches) - 1):
        start = batches[i]
        end = batches[i + 1]
        batch_data = user_item_rating[start:end, :]
        loss = _getLossUpdateDerivative(batch_data, weights, biases,
                                        dWeights, dBiases, NN, modelArg)
        cost += loss

        if not modelArg.regularize_bias:
            weight_decay = 0.5 *\
                reduce(
                    lambda x, y: x + y, map(lambda z:
                                            np.power(
                                                weights[z], 2).sum() *
                                            modelArg.lamda[z],
                                            range(len(weights))))
        else:
            weight_decay = 0.5 *\
                reduce(
                    lambda x, y: x + y, map(lambda z:
                                            np.power(
                                                weights[z], 2).sum() *
                                            modelArg.lamda[z],
                                            range(len(weights))))
            weight_decay = 0.5 *\
                reduce(
                    lambda x, y: x + y, map(lambda z:
                                            np.power(
                                                biases[z], 2).sum() *
                                            modelArg.lamda[z],
                                            range(len(biases))))
        cost += weight_decay

        for i in range(len(dWeights)):
            # dWeights[i] += modelArg.lamda * weights[i]
            dWeights[i] += modelArg.lamda[i] * weights[i]

        if modelArg.regularize_bias:
            for i in range(len(dBiases)):
                # dBiases[i] += modelArg.lamda * biases[i]
                dBiases[i] = dBiases[i].reshape(dBiases_old[i].shape)
                dBiases[i] += modelArg.lamda[i] * biases[i]

        for i in range(len(weights)):
            temp_wderiv = (
                alpha * dWeights[i] + dWeights_old[i] * modelArg.momentum)
            weights[i] -= temp_wderiv
            dWeights_old[i] = temp_wderiv

        for i in range(len(biases)):
            temp_bderiv = (
                alpha * dBiases[i] + dBiases_old[i] * modelArg.momentum)
            biases[i] -= temp_bderiv
            dBiases_old[i] = temp_bderiv
    return dWeights_old, dBiases_old


def updateAdagrad(user_item_rating, NN, modelArg, counter, batch_size,
                  alpha, dWeights_old, dBiases_old):
    counter.increment()

    # user_item_rating = shuffle(user_item_rating)
    weights = NN.weights
    biases = NN.biases

    dWeights = []
    for weight in weights:
        dWeights.append(np.zeros(shape=weight.shape))

    dBiases = []
    for bias in biases:
        dBiases.append(np.zeros(shape=bias.shape))

    ##################################### Batch loop #########################

    m, n = user_item_rating.shape
    batches = range(0, m, batch_size)
    if batches[-1] != m:
        batches.append(m)
    cost = 0.0
    for i in range(len(batches) - 1):
        start = batches[i]
        end = batches[i + 1]
        batch_data = user_item_rating[start:end, :]
        loss = _getLossUpdateDerivative(batch_data, weights, biases,
                                        dWeights, dBiases, NN, modelArg)
        cost += loss
        if not modelArg.regularize_bias:
            weight_decay = 0.5 *\
                reduce(
                    lambda x, y: x + y, map(lambda z:
                                            np.power(
                                                weights[z], 2).sum() *
                                            modelArg.lamda[z],
                                            range(len(weights))))
        else:
            weight_decay = 0.5 *\
                reduce(
                    lambda x, y: x + y, map(lambda z:
                                            np.power(
                                                weights[z], 2).sum() *
                                            modelArg.lamda[z],
                                            range(len(weights))))
            weight_decay = 0.5 *\
                reduce(
                    lambda x, y: x + y, map(lambda z:
                                            np.power(
                                                biases[z], 2).sum() *
                                            modelArg.lamda[z],
                                            range(len(biases))))
        cost += weight_decay

        for i in range(len(dWeights)):
            # dWeights[i] += modelArg.lamda * weights[i]
            dWeights[i] += modelArg.lamda[i] * weights[i]

        if modelArg.regularize_bias:
            for i in range(len(dBiases)):
                # dBiases[i] += modelArg.lamda * biases[i]
                dBiases[i] = dBiases[i].reshape(dBiases_old[i].shape)
                dBiases[i] += modelArg.lamda[i] * biases[i]

        if counter.count == 1:
            dWeights_old[i] += np.power(dWeights[i], 2)
            dBiases_old[i] += np.power(dBiases[i], 2)
            continue

        for i in range(len(weights)):
            temp_wderiv = np.divide(
                dWeights[i], np.sqrt(dWeights_old[i] + 1)) * alpha
            weights[i] -= temp_wderiv
            dWeights_old[i] += np.power(dWeights[i], 2)

        for i in range(len(biases)):
            temp_bderiv = np.divide(
                dBiases[i], np.sqrt(dBiases_old[i]) + 1) * alpha
            biases[i] -= temp_bderiv
            dBiases_old[i] += np.power(dBiases[i], 2)

    return dWeights_old, dBiases_old


def trainSGD(train, test, num_iter, evaluate, weights, biases, learn_rate, modelArg, NN, counter, batch_size, driver=False):
    old_rmse = float("inf")
    dWeights_old = []
    for weight in weights:
        dWeights_old.append(np.zeros(shape=weight.shape))

    dBiases_old = []
    for bias in biases:
        dBiases_old.append(np.zeros(shape=bias.shape))

    for i in range(num_iter):
        # t = shuffle(train)
        t = train
        dWeights_old, dBiases_old = updateSGD(t, NN, modelArg, counter,
                                              batch_size, learn_rate,
                                              dWeights_old, dBiases_old)

        if (i % 5 == 0):
            trmse, tmae = evaluate.calculateRMSEandMAE(train, test)
            rmse, mae = evaluate.calculateRMSEandMAE(train, train)
            sign = "+" if rmse < old_rmse else "-"
            print "Fold :%d Test RMSE: %f MAE: %f \t %s" % (i, trmse, tmae, sign)

            # print "Fold :%d Train RMSE: %f MAE: %f" % (i, rmse, mae)
            if driver:
                if rmse < old_rmse:
                    bestWeights = deepcopy(NN.weights)
                    bestBiases = deepcopy(NN.biases)
                    learn_rate *= 1.01
                    old_rmse = rmse
                elif rmse > old_rmse:
                    NN.weights = bestWeights
                    NN.biases = bestBiases
                    print "Reducing learning rate"
                    learn_rate *= 0.5

        if learn_rate < EPS:
            break


def trainAdagrad(train, test, num_iter, evaluate, weights, biases, learn_rate, modelArg, NN, counter, batch_size, driver=False):
    old_rmse = float("inf")
    dWeights_old = []
    for weight in weights:
        dWeights_old.append(np.zeros(shape=weight.shape))

    dBiases_old = []
    for bias in biases:
        dBiases_old.append(np.zeros(shape=bias.shape))

    for i in range(num_iter):
        # t = shuffle(train)
        t = trian
        dWeights_old, dBiases_old = updateAdagrad(t, NN, modelArg, counter,
                                                  batch_size, learn_rate,
                                                  dWeights_old, dBiases_old)
        if (i % 5 == 0):
            rmse, mae = evaluate.calculateRMSEandMAE(train, test)
            print "Fold :%d Test RMSE: %f MAE: %f" % (i, rmse, mae)
