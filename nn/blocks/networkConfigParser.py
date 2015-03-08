import yaml
from nn import *
from activations import *
from exceptions import Exception
from utils import *
# from cfRBM import ModelArgs


class NetworkConfigParser(object):

    @classmethod
    def getDataInfo(cls, path):
        with open(path) as fp:
            data = yaml.load(fp)
            data_info = data["data"]
            train_path = data_info["train"]
            test_path = data_info["test"]
            save_path = data_info["save"]
        return (train_path, test_path, save_path)

    @classmethod
    def constructModelArgs(cls, path, ModelArgs):
        kwargs = {}
        with open(path) as fp:
            data = yaml.load(fp)
            params = data["params"]
            if "reg_bias" in params:
                kwargs["regularize_bias"] = params["reg_bias"]
            if "momentum" in params:
                kwargs["momentum"] = params["momentum"]
            if "mean" in params:
                kwargs["mean"] = params["mean"]
            if "beta" in params:
                kwargs["beta"] = params["beta"]
            if "mean_normalization" in params:
                kwargs["mean"] = params["mean_normalization"]
            if "learn_rate" in params:
                kwargs["learn_rate"] = params["learn_rate"]
            kwargs["lamda"] = params["lamda"]
            kwargs["max_iter"] = params["max_iter"]
            if "optimizer" in params:
                kwargs["optimizer"] = params["optimizer"]
            if "batch_size" in params:
                kwargs["batch_size"] = params["batch_size"]
        args = ModelArgs(**kwargs)
        return args

    @classmethod
    def constructNetwork(cls, path):
        nn = NN()
        with open(path) as fp:
            data = yaml.load(fp)
            layers = data["layers"]
            layer_ids = layers.keys()
            layer_ids.sort()
            for layer_id in layer_ids:
                layer_info = layers[layer_id]
                layer = cls._constructLayer(layer_info)
                nn.addLayer(layer)
        nn.finalize()
        return nn

    @classmethod
    def _constructLayer(cls, layer_info):
        num_nodes = layer_info["num_nodes"]
        activation = layer_info["activation"].lower()

        if "partial" in layer_info:
            isPartial = layer_info["partial"]
        else:
            isPartial = False

        if "dropout" in layer_info:
            dropout = layer_info["dropout"]
        else:
            dropout = 0.0

        if "sparsity" in layer_info:
            sparsity = layer_info["sparsity"]
        else:
            sparsity = None

        if "binary" in layer_info:
            binary = layer_info["binary"]
        else:
            binary = False

        layer_type = layer_info["type"].lower()
        activation = cls._getActivation(activation)
        ltype = cls._getLayerType(layer_type)

        layer = Layer(num_nodes, activation, ltype)
        if isPartial:
            layer.setPartial()
        if dropout:
            layer.setDropout(dropout)
        if sparsity:
            layer.setSparsity(sparsity)
        if binary:
            layer.setBinary()
        return layer

    @classmethod
    def _getLayerType(cls, layer_type):
        if layer_type == "input":
            return LayerType.INPUT
        elif layer_type == "hidden":
            return LayerType.HIDDEN
        elif layer_type == "output":
            return LayerType.OUTPUT
        else:
            raise Exception("Unknown Layer Type")

    @classmethod
    def _getActivation(cls, activation):
        if activation == "sigmoid":
            return Sigmoid()
        elif activation == "identity":
            return Identity()
        elif activation == "relu":
            return RELU()
        elif activation == "nrelu":
            return NRELU()
        elif activation == "tanh":
            return Tanh()
        else:
            raise Exception("Unknown Activation Function")

    @classmethod
    def validateNetwork(cls, network, modelArgs):
        pass

if __name__ == '__main__':
    # parser = NetworkConfigParser()
    data_info = NetworkConfigParser.getDataInfo("config/net.yaml")
    modelArgs = NetworkConfigParser.constructModelArgs("config/net.yaml")
    nn = NetworkConfigParser.constructNetwork("config/net.yaml")
    print nn, modelArgs, data_info
