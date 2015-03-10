from ae import AE
import numpy as np
from nn.blocks.networkConfigParser import NetworkConfigParser
from dataUtils.data import Data, loadTestData
from utils.metrics.evaluate import EvaluateNN
from ae_utils import Counter, ModelArgs


def loadModel(config_path):
    modelArgs = NetworkConfigParser.constructModelArgs(config_path, ModelArgs)
    nn = NetworkConfigParser.constructNetwork(config_path)
    train_path, test_path, save_path = NetworkConfigParser.getDataInfo(
        config_path)
    ae = AE(nn, modelArgs)
    theta = np.load(save_path + ".npy")
    ae.setParameters(theta)
    return ae


def loadData(config_path):
    train_path, test_path, save_path = NetworkConfigParser.getDataInfo(
        config_path)
    nn = NetworkConfigParser.constructNetwork(config_path)
    d = Data()
    d.import_ratings(train_path, shape=(None, nn.layers[0].num_units))
    train = d.R.copy()
    test = loadTestData(d, test_path)
    return train, test


def LoadDataAndMapping(config_path):
    train_path, test_path, save_path = NetworkConfigParser.getDataInfo(
        config_path)
    nn = NetworkConfigParser.constructNetwork(config_path)
    d = Data()
    d.import_ratings(train_path, shape=(None, nn.layers[0].num_units))
    train = d.R.copy()
    test = loadTestData(d, test_path)
    usermap = {v: k for k, v in d.users.items()}
    itemmap = {v: k for k, v in d.items.items()}
    return train, test, usermap, itemmap


# def evaluateFolds(config_path, nfolds):
#     rmses = []
#     maes = []
#     for i in range(1, nfolds + 1):
#         model = loadModel(config_path)
#         train, test = loadData(config_path)
#         evaluate = EvaluateNN(model)
#         rmse, mae = evaluate.calculateRMSEandMAE(train, test)
#         rmses.append(rmse)
#         maes.append(mae)
#     return rmses, maes

# if __name__ == '__main__':
#     import argparse
#     from utils.statUtil import getMeanCI
#     parser = argparse.ArgumentParser(description='Description')
#     parser.add_argument(
#         '--config', '-c', help='configuration file', required=True)
#     parser.add_argument(
#         '--nfold', '-n', help='number of folds ', required=True)
#     args = parser.parse_args()
#     nfolds = int(args.nfold)
#     config_path = args.config

#     rmses, maes = evaluateFolds(config_path, nfolds)
#     ci_rmse = getMeanCI(rmses, 0.95)
#     ci_mae = getMeanCI(maes, 0.95)
#     print ci_rmse
#     print ci_mae
