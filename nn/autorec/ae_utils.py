class Counter(object):

    """docstring for Counter"""

    def __init__(self):
        super(Counter, self).__init__()
        self.count = 0

    def increment(self):
        self.count += 1


class ModelArgs(object):

    """docstring for ModelArgs"""

    def __init__(self, learn_rate=0.001, lamda=1.0, regularize_bias=True,
                 isDenoising=False, noisePercent=0.0, beta=None, momentum=0.0,
                 num_threads=16, mean=0.0, max_iter=200, optimizer=None,
                 batch_size=20000):
        super(ModelArgs, self).__init__()
        self.learn_rate = learn_rate
        self.lamda = lamda
        self.regularize_bias = regularize_bias
        self.isDenoising = isDenoising
        self.noisePercent = noisePercent
        self.beta = beta
        self.momentum = momentum
        self.num_threads = num_threads
        self.mean = mean
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.batch_size = batch_size

    def __str__(self):
        string = ""
        for key in self.__dict__.keys():
            string += "%s: %s\t" % (key, str(self.__dict__[key]))
        return string
