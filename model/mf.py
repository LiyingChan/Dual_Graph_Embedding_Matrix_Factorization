import numpy as np
import matplotlib.pylab as plt
from collections import defaultdict
from metrics.metric import Metric
from utility.tools import denormalize
from reader.rating import RatingGetter
from configx.configx import ConfigX
import pickle
class MF(object):
    """
    docstring for MF
    the base class for matrix factorization based model-parent class

    """

    def __init__(self):
        super(MF, self).__init__()
        self.config = ConfigX()
        self.rg = RatingGetter()  # loading raing data
        # self.init_model()
        self.iter_rmse = []
        self.iter_mae = []
        pass

    def init_model(self):
        self.P = np.random.rand(self.rg.get_train_size()[0], self.config.factor) / (
        self.config.factor ** 0.5)  # latent user matrix
        self.Q = np.random.rand(self.rg.get_train_size()[1], self.config.factor) / (
        self.config.factor ** 0.5)  # latent item matrix
        self.Bu = np.random.rand(self.rg.get_train_size()[0])  # bias value of user
        self.Bi = np.random.rand(self.rg.get_train_size()[1])  # bais value of item
        print(self.rg.get_train_size()[0])
        print(self.rg.get_train_size()[1])
        self.loss, self.lastLoss = 0.0, 0.0
        self.lastRmse, self.lastMae = 10.0, 10.0
        pass

    def train_model(self):
        pass

    def valid_model(self):
        res = []
        for ind, entry in enumerate(self.rg.validSet()):
            user, item, rating = entry
            # predict
            prediction = self.predict(user, item)
            # denormalize
            prediction = denormalize(prediction, self.config.min_val, self.config.max_val)

            pred = self.checkRatingBoundary(prediction)
            # add prediction in order to measure
            # self.dao.testData[ind].append(pred)
            res.append([user, item, rating, pred])
        rmse = Metric.RMSE(res)
        mae = Metric.MAE(res)
        self.iter_rmse.append(rmse)  # for plot
        self.iter_mae.append(mae)
        return rmse, mae

    # test all users in test set
    def predict_model(self):
        res = []
        for ind, entry in enumerate(self.rg.testSet()):
            user, item, rating = entry
            # predict
            prediction = self.predict(user, item)
            # denormalize
            prediction = denormalize(prediction, self.config.min_val, self.config.max_val)

            pred = self.checkRatingBoundary(prediction)
            # add prediction in order to measure
            # self.dao.testData[ind].append(pred)
            res.append([user, item, rating, pred])
        rmse = Metric.RMSE(res)
        mae = Metric.MAE(res)
        print('learning_Rate = %.5f rmse=%.5f mae=%.5f' % \
              (self.config.lr, rmse, mae))
        # self.iter_rmse.append(rmse)  # for plot
        # self.iter_mae.append(mae)
        return rmse, mae

    # test cold start users among test set
    def predict_model_cold_users(self):
        res = []
        for user in self.rg.testColdUserSet_u.keys():
            for item in self.rg.testColdUserSet_u[user].keys():
                rating = self.rg.testColdUserSet_u[user][item]
                pred = self.predict(user, item)
                # denormalize
                pred = denormalize(pred, self.config.min_val, self.config.max_val)
                pred = self.checkRatingBoundary(pred)
                res.append([user, item, rating, pred])
        rmse = Metric.RMSE(res)
        return rmse

    def predict(self, u, i):
        if self.rg.containsUser(u) and self.rg.containsItem(i):
            u = self.rg.user[u]
            i = self.rg.item[i]
            return self.P[u].dot(self.Q[i]) + self.rg.globalMean + self.Bi[i] + self.Bu[u]
        elif self.rg.containsUser(u) and not self.rg.containsItem(i):
            return self.rg.userMeans[u]
        elif not self.rg.containsUser(u) and self.rg.containsItem(i):
            return self.rg.itemMeans[i]
        else:
            return self.rg.globalMean

    def checkRatingBoundary(self, prediction):
        if prediction > self.config.max_val:
            return self.config.max_val
        elif prediction < self.config.min_val:
            return self.config.min_val
        else:
            return round(prediction, 3)

    def save_P(self, file_name):
        P_dic = defaultdict(list)
        with open(file_name, 'wb') as f:
            print(self.rg.get_train_size()[0])
            for i in range(self.rg.get_train_size()[0]):
                if i in self.rg.id2user.keys():
                    user = self.rg.id2user[i]
                    P_dic[user] = self.P[i]
            pickle.dump(P_dic, f)

    def save_Q(self, file_name):
        Q_dic = defaultdict(list)
        with open(file_name, 'wb') as f:
            print(self.rg.get_train_size()[1])
            for i in range(self.rg.get_train_size()[1]):
                if i in self.rg.id2item.keys():
                    user = self.rg.id2item[i]
                    Q_dic[user] = self.Q[i]
            pickle.dump(Q_dic, f)

    def isConverged(self, iter):
        from math import isnan
        if isnan(self.loss):
            print(
                'Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!')
            exit(-1)
        # measure = self.performance()
        # value = [item.strip()for item in measure]
        # with open(self.algorName+' iteration.txt')
        deltaLoss = (self.lastLoss - self.loss)
        rmse, mae = self.valid_model()

        # early stopping
        if self.config.isEarlyStopping == True:
            cond = self.lastRmse < rmse
            if cond:
                print('test rmse increase, so early stopping')
                # P_path = '../data/P/P_path_100_feas.pkl'
                # self.save_P(P_path)
                # Q_path = '../data/Q/Q_300_feas.pkl'
                # self.save_Q(Q_path)
                return cond
            self.lastRmse = rmse
            self.lastMae = mae
        print('%s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f rmse=%.5f mae=%.5f' % \
              (self.__class__, iter, self.loss, deltaLoss, self.config.lr, rmse, mae))
        # check if converged
        cond = abs(deltaLoss) < self.config.threshold
        converged = cond
        # if not converged:
        # 	self.updateLearningRate(iter)
        self.lastLoss = self.loss
        # shuffle(self.dao.trainingData)
        return converged

    def updateLearningRate(self, iter):
        if iter > 1:
            if abs(self.lastLoss) > abs(self.loss):
                self.config.lr *= 1.05
            else:
                self.config.lr *= 0.5
        if self.config.lr > 1:
            self.config.lr = 1

    def show_rmse(self):
        '''
        show figure for rmse and epoch
        '''
        nums = range(len(self.iter_rmse))
        plt.plot(nums, self.iter_rmse, label='RMSE')
        plt.plot(nums, self.iter_mae, label='MAE')
        plt.xlabel('# of epoch')
        plt.ylabel('metric')
        plt.title(self.__class__)
        plt.legend()
        plt.show()
        pass
