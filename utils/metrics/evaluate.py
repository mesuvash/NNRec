
from math import fabs, sqrt
import numpy as np
import scipy.sparse


class Evaluate(object):

    """docstring for Evaluate"""

    def __init__(self, predictor):
        super(Evaluate, self).__init__()
        self.predictor = predictor

    def calculateRMSEandMAE(self, test):
        pass


class EvaluateConstant(Evaluate):

    def __init__(self, predictor):
        super(Evaluate, self).__init__()
        self.predictor = predictor

    def calculateRMSEandMAE(self, test, ):
        rmse = 0.0
        mae = 0.0
        count = 0
        for user in test:
            ratings = test[user]["ratings"]
            for actual in ratings:
                predicted = self.predictor.predict(user, 1)
                rmse += (actual - predicted) ** 2
                mae += fabs(actual - predicted)
                count += 1
        return [sqrt(rmse / count), mae / count]


class EvaluateNN(Evaluate):

    """docstring for EvaluateRBM"""

    def __init__(self, predictor, scale=1.0, default=3.0):
        super(EvaluateNN, self).__init__(predictor)
        self.scale = scale
        self.default = default

    def calculateRMSEandMAE(self, train, test, cold=None):
        predictions = self.predictor.predict(train, test)
        if scipy.sparse.isspmatrix(train):
            predictions.data = predictions.data * self.scale
            err = np.fabs(predictions.data - test.data * self.scale)
            total_instances = len(test.data)
        else:
            err = np.fabs(predictions - test * self.scale)
            total_instances = test.size
        cold_err = []
        if cold is not None:
            cold_err = map(lambda x: np.fabs(x - self.default), cold)
            total_instances += len(cold)
        cold_err = np.array(cold_err)

        rmse = np.sqrt(
            (np.power(err, 2).sum() + np.power(cold_err, 2).sum()) / (total_instances))
        mae = (err.sum() + cold_err.sum()) / total_instances
        return [rmse, mae]


class EvaluateRBM(EvaluateNN):

    """docstring for EvaluateRBM"""

    def __init__(self, predictor, scale=1.0, default=3.0):
        super(EvaluateRBM, self).__init__(predictor, scale, default)

    def calculateRMSEandMAE(self, btrain, btest, test,
                            cold_ratings=None, default_rating=3.0):
        predictions = self.predictor.predict(btrain, btest)
        if scipy.sparse.isspmatrix(btrain):
            predictions = predictions * self.scale
            err = np.fabs(predictions - test.data)
            total_instances = len(test.data)
        else:
            err = np.fabs(predictions - test)
            total_instances = test.size
        cold_err = []
        if cold_ratings:
            for rating in cold_ratings:
                cold_err.append(np.fabs(rating - default_rating))
            total_instances += len(cold_err)
        cold_err = np.array(cold_err)
        # print(np.power(err, 2).sum() + np.power(cold_err, 2).sum())
        rmse = np.sqrt((np.power(err, 2).sum() +
                        np.power(cold_err, 2).sum()) / total_instances)
        mae = (err.sum() + cold_err.sum()) / total_instances
        return [rmse, mae]


class EvaluateJointAE(Evaluate):

    """docstring for EvaluateRBM"""

    def __init__(self, predictor, default=3.0):
        super(EvaluateJointAE, self).__init__(predictor)

    def calculateRMSEandMAE(self, test, ufeats, ifeats,
                            cold_ratings=None, default_rating=3.0):
        mae = 0.0
        rmse = 0.0
        for i in range(test.shape[1]):
            user, item, rating = test[:, i]
            prediction = self.predictor.predict(
                int(user), int(item), ufeats, ifeats)
            error = np.fabs(rating - prediction)
            mae += error
            rmse += np.power(error, 2)
        cold_err = []
        if cold_ratings:
            for rating in cold_ratings:
                cold_err.append(np.fabs(rating - default_rating))
        cold_err = np.array(cold_err)
        total_instances = test.shape[1] + len(cold_err)

        mae = (mae + cold_err.sum()) / total_instances
        rmse = np.sqrt((rmse + np.power(cold_err, 2).sum()) / total_instances)
        return [rmse, mae]


class EvaluateJointAE1(Evaluate):

    """docstring for EvaluateRBM"""

    def __init__(self, predictor, default=3.0):
        super(EvaluateJointAE1, self).__init__(predictor)

    def calculateRMSEandMAE(self, test, ufeats, ifeats, ufeats1, ifeats1,
                            cold_ratings=None, default_rating=3.0):
        mae = 0.0
        rmse = 0.0
        for i in range(test.shape[1]):
            user, item, rating = test[:, i]
            prediction = self.predictor.predict(
                int(user), int(item), ufeats, ifeats, ufeats1, ifeats1)
            error = np.fabs(rating - prediction)
            mae += error
            rmse += np.power(error, 2)
        cold_err = []
        if cold_ratings:
            for rating in cold_ratings:
                cold_err.append(np.fabs(rating - default_rating))
        cold_err = np.array(cold_err)
        total_instances = test.shape[1] + len(cold_err)

        mae = (mae + cold_err.sum()) / total_instances
        rmse = np.sqrt((rmse + np.power(cold_err, 2).sum()) / total_instances)
        return [rmse, mae]
