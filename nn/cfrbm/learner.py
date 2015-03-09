from cfRBM import *
from dataUtils.data import loadTrainTest
# from nn.blocks.nn import Layer, NN, LayerType
from nn.blocks.activations import *
from nn.blocks.networkConfigParser import NetworkConfigParser
import yaml


def train(config_path):
    configparser = NetworkConfigParser()
    nn = configparser.constructNetwork(config_path)
    modelArgs = configparser.constructModelArgs(config_path, ModelArgs)
    train_path, test_path, save_path = configparser.getDataInfo(config_path)
    print nn

    data = yaml.load(open(config_path))
    params = data["params"]
    k = params["k"]

    n_vis = int(nn.layers[0].num_units / k)
    train, test, cold_ratings = loadTrainTest(train_path, test_path,
                                              shape=(None, n_vis))

    min_rating, max_rating = train.data.min(), train.data.max()
    increment = 1
    mapping = dict(zip(np.arange(min_rating, max_rating + increment,
                                 increment), np.arange(k)))
    modelArgs.mapping = mapping
    modelArgs.k = k
    bintrain = binarizeSparseMatrix(train, k, mapping)
    bintest = binarizeSparseMatrix(test, k, mapping)
    del train
    model = RBM(nn, modelArgs)
    optimizer = RbmOptimizer(model)
    optimizer.minibatchTrain(bintrain, bintest, test, modelArgs.batch_size)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument(
        '--config', '-c', help='configuration file', required=True)
    args = parser.parse_args()
    config_path = args.config
    train(config_path)
