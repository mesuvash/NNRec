from utils.metrics.evaluate import EvaluateNN
from nn.blocks.networkConfigParser import NetworkConfigParser
from lossDeriv import *
from dataUtils.data import loadTrainTest
from ae import AE
from optimizers import getOptimizer
from ae_utils import Counter, ModelArgs


def train(config_path):
    modelArgs = NetworkConfigParser.constructModelArgs(config_path, ModelArgs)
    nn = NetworkConfigParser.constructNetwork(config_path)
    train_path, test_path, save_path = NetworkConfigParser.getDataInfo(
        config_path)
    print nn
    # TODO : Arguments
    num_hid = nn.layers[1].num_units
    shape = (None, nn.layers[0].num_units)
    train, test, cold = loadTrainTest(train_path, test_path,
                                      shape=shape)

    ae = AE(nn, modelArgs)
    evaluate = EvaluateNN(ae)
    theta = ae.nn.getFlattenParams()
    ae.setParameters(theta)
    iterCounter = Counter()
    optimizer = getOptimizer(modelArgs.optimizer, ae, evaluate, theta,
                             train, test, nn, modelArgs, iterCounter,
                             modelArgs.batch_size,
                             modelArgs.max_iter[0])

    optimizer.step_grow = 5.0
    k = 0
    for info in optimizer:
        print "Iteration %d" % k
        if k == 5:
            optimizer.step_grow = 1.2
        if k % 5 == 0:
            ae.setParameters(theta)
            rmse, mae = evaluate.calculateRMSEandMAE(train, test)
            print "Fold :%d Final Train RMSE: %f Train MAE: %f" % (i,
                                                                   rmse, mae)
        if k > modelArgs.max_iter[0]:
            break
        k += 1
    if save_path:
        _theta = ae.getParameters()
        np.save(save_path, _theta)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument(
        '--config', '-c', help='configuration file', required=True)
    args = parser.parse_args()
    config_path = args.config
    i = 1
    train(config_path)
