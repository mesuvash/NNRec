from AE import AE
import numpy as np
from networkConfigParser import NetworkConfigParser
from statUtil import getMeanCI
from data import Data, loadTestData
from evaluate import EvaluateNN


def loadModel(config_path, i):
    modelArgs = NetworkConfigParser.constructModelArgs(config_path)
    nn = NetworkConfigParser.constructNetwork(config_path)
    train_path, test_path, save_path = NetworkConfigParser.getDataInfo(
        config_path)
    num_hid = nn.layers[1].num_units
    save_path = save_path % (num_hid, modelArgs.lamda[0], i)
    ae = AE(nn, modelArgs)
    theta = np.load(save_path + ".npy")
    ae.setParameters(theta)
    return ae


def loadData(config_path, i):
    train_path, test_path, save_path = NetworkConfigParser.getDataInfo(
        config_path)
    nn = NetworkConfigParser.constructNetwork(config_path)
    d = Data()
    d.import_ratings(train_path % i, shape=(None, nn.layers[0].num_units))
    train = d.R.copy()
    test = loadTestData(d, test_path % i)
    return train, test


def evaluateFolds(config_path, nfolds):
    rmses = []
    maes = []
    for i in range(1, nfolds + 1):
        model = loadModel(config_path, i)
        train, test = loadData(config_path, i)
        evaluate = EvaluateNN(model)
        rmse, mae = evaluate.calculateRMSEandMAE(train, test)
        rmses.append(rmse)
        maes.append(mae)
    return rmses, maes

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument(
        '--config', '-c', help='configuration file', required=True)
    parser.add_argument(
        '--nfold', '-n', help='number of folds ', required=True)
    args = parser.parse_args()
    nfolds = int(args.nfold)
    config_path = args.config

    rmses, maes = evaluateFolds(config_path, nfolds)
    ci_rmse = getMeanCI(rmses, 0.95)
    ci_mae = getMeanCI(maes, 0.95)
    print ci_rmse
    print ci_mae
