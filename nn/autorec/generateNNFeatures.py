import numpy as np
from nn.blocks.networkConfigParser import NetworkConfigParser
from modelLoader import loadModel, LoadDataAndMapping


def dumpArray(array, outpath, mapping):
    fp = open(outpath, "wb")
    m, n = array.shape
    for i in range(m):
        for j in range(n):
            value = array[i, j]
            if value != 0:
                fp.write("%s\t%d\t%f\n" % (mapping[i], j, array[i, j]))
    fp.close()


def dumpFeatures(config_path, mtype, outpath):
    model = loadModel(config_path)
    train, test, usermap, itemmap = LoadDataAndMapping(config_path)
    target_layer = 1
    targetLayerData = model.getActivationOfLayer(train, target_layer)
    if mtype == "user":
        dumpArray(targetLayerData, outpath, usermap)
    if mtype == "item":
        dumpArray(targetLayerData, outpath, itemmap)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument(
        '--config', '-c', help='configuration file', required=True)
    parser.add_argument(
        '--mtype', '-m', help='configuration file', required=True)
    parser.add_argument(
        '--outfile', '-o', help='configuration file', required=True)
    args = parser.parse_args()
    config_path = args.config
    mtype = args.mtype
    outfile = args.outfile
    dumpFeatures(config_path, mtype, outfile)
