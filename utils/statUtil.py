from scipy import stats
import math


def getConfidenceInterval(data, percent, distribution="t"):
    n, min_max, mean, var, skew, kurt = stats.describe(data)
    std = math.sqrt(var)
    if distribution == "t":
        R = stats.t.interval(
            percent, len(data) - 1, loc=mean, scale=std / math.sqrt(len(data)))
    else:
        R = stats.norm.interval(
            percent, loc=mean, scale=std / math.sqrt(len(data)))
    return mean, R


def getMeanCI(data, percent, distribution="t"):
    mean, errors = getConfidenceInterval(data, percent)
    return mean, (errors[1] - errors[0]) / 2.0

if __name__ == '__main__':
    import numpy as np
    s = np.array([3, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 6])
    print getConfidenceInterval(s, 0.95)
