# encoding:utf-8
import sys

sys.path.append("..")
import numpy as np
from collections import defaultdict
from random import randint
from random import shuffle,choice
from math import log
import gensim.models.word2vec as w2v
from prettyprinter import cpprint

from mf import MF
# from reader.trust import TrustGetter
from utility.matrix import SimMatrix
from utility.similarity import cosine
from utility import util


class CUNE(MF):
    """
    docstring for CUNE

    Zhang et al. Collaborative User Network Embedding for Social Recommender Systems. SDM
    """

    def __init__(self):
        super(CUNE, self).__init__()
        self.config.lambdaP = 0.01
        self.config.lambdaQ = 0.01
        self.config.alpha = 0.01
        self.config.isEarlyStopping = True
        # self.tg = TrustGetter()
        self.config.walkCount = 30
        self.config.walkLength = 20
        self.config.walkDim = 20
        self.config.winSize = 5
        self.config.topK = 50

    def init_model(self, k):
        super(CUNE, self).init_model(k)
        self.user_sim = SimMatrix()
        self.generate_cu_net() # 构建uu网络
        self.deep_walk()

        print('Constructing similarity matrix...')
        # self.W = np.zeros((self.rg.get_train_size()[0], self.config.walkDim))
        self.topKSim = defaultdict(dict)
        i = 0
        for user1 in self.CUNet:
            sims = {}
            for user2 in self.CUNet:
                if user1 != user2:
                    wu1 = self.model[str(user1)] # 取出embedding
                    wu2 = self.model[str(user2)]
                    sims[user2]=cosine(wu1,wu2) # 计算uu相似性
            self.topKSim[user1] = sorted(sims.items(), key=lambda d: d[1], reverse=True)[:self.config.topK] # 按照value来排序，{u1:{u2:1.0, ...}, ...}每个user的键值是该user的k个最相似好友
            i += 1
            if i % 200 == 0:
                print('progress:', i, '/', len(self.CUNet)) # 200个user为一组输出进度
        # print(self.topKSim)
        #构建被关注列表
        print('Constructing desimilarity matrix...')
        self.topKSimBy = defaultdict(dict)
        for user in self.topKSim:
            users = self.topKSim[user]
            for user2 in users: # user的相似好友中
                self.topKSimBy[user2[0]][user] = user2[1] # 把“关注字典”的key和value互换，得到“被关注字典”：{u2:{u1:1.0, ...}, ...}
        print('Similarity matrix finished.')


    def generate_cu_net(self):
        print('Building collaborative user network...')
        itemNet = {}
        for item in self.rg.trainSet_i: # 外层key为item 里层key为user value为评分
            if len(self.rg.trainSet_i[item])>1: # 如果item被一个以上的user评过分
                itemNet[item] = self.rg.trainSet_i[item] # 把这些item的里层字典赋值给itemNet

        filteredRatings = defaultdict(list)
        for item in itemNet:
            for user in itemNet[item]:
                if itemNet[item][user] > 0:
                    filteredRatings[user].append(item) # 若user和item有交互，把 item append给filteredRatings[user]

        self.CUNet = defaultdict(list)
        for user1 in filteredRatings:
            s1 = set(filteredRatings[user1]) # s1是与user1有交互的物品集
            for user2 in filteredRatings:
                if user1 != user2: # 遍历filteredRatings所有user，如果不是同一个user
                    s2 = set(filteredRatings[user2]) # s2是与user2有交互的物品集
                    weight = len(s1.intersection(s2)) # 把两个不同user的评过分的物品集的交集的物品个数作为uu网络的uu权重
                    if weight > 0:
                        self.CUNet[user1] += [user2] # 得到{user1:[user2,......]}
        # cpprint(self.CUNet)
        pass
    def deep_walk(self):
        print('Generating random deep walks...')
        self.walks = []
        self.visited = defaultdict(dict)
        for user in self.CUNet:
            for t in range(self.config.walkCount):
                path = [str(user)]
                lastNode = user
                for i in range(1,self.config.walkLength):
                    nextNode = choice(self.CUNet[lastNode]) # 从user的交互用户中随机选取下一个顶点
                    count=0
                    while(nextNode in self.visited[lastNode]): # 当该顶点已经访问过了，则重新选取
                        nextNode = choice(self.CUNet[lastNode])
                        #break infinite loop
                        count+=1
                        if count==self.config.walkLength: # 随机游走长度为walkLength
                            break
                    path.append(str(nextNode)) # 构建访问序列
                    self.visited[user][nextNode] = 1 # 得到{user:{nextNode:1, ...}, ...}
                    lastNode = nextNode # 把nextNode作为当前顶点
                self.walks.append(path) # walk = [[访问过的顶点序列], [...], ...]
                #print path
        # shuffle(self.walks)
        # cpprint(self.walks)
        print('Generating user embedding...')
        self.model = w2v.Word2Vec(self.walks, size=self.config.walkDim, window=5, min_count=0, iter=3)
        # 对所有访问序列作word2vec，访问序列是我们要分析的语料；size: 词向量的维度；window：即词向量上下文最大距离；min_count:需要计算词向量的最小词频；iter: 随机梯度下降法中迭代的最大次数
        print('User embedding generated.')
        pass

    def train_model(self, k):
        super(CUNE, self).train_model(k)
        iteration = 0
        while iteration < self.config.maxIter:
            self.loss = 0
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line
                u = self.rg.user[user]
                i = self.rg.item[item]
                error = rating - self.predict(user, item)
                self.loss += 0.5 * error ** 2
                p, q = self.P[u], self.Q[i]# p表示user本身的隐向量

                social_term_p, social_term_loss = np.zeros((self.config.factor)), 0.0
                followees = self.topKSim[user] #self.tg.get_followees(user) #self.topKSim[user]
                # print(followees)
                for followee in followees: # 遍历user关注的用户
                    if self.rg.containsUser(followee[0]): # 如果user关注的用户在训练集中
                        # s = self.user_sim[user][followee]
                        uf = self.P[self.rg.user[followee[0]]] # 把followee[0]序号映射为self.rg.user[followee[0]]的id。uf表示user关注的好友的隐向量
                        social_term_p += followee[1]* (p - uf)  # 用在GD
                        social_term_loss += followee[1]* ((p - uf).dot(p - uf))

                social_term_m = np.zeros((self.config.factor))
                followers = self.topKSimBy[user]
                followers = sorted(followers.items(), key=lambda d: d[1], reverse=True)[:self.config.topK] # 关注user的用户中取出k个最相似的
                for follower in followers: # 遍历关注user的用户
                    if self.rg.containsUser(follower[0]):
                        ug = self.P[self.rg.user[follower[0]]] # uf表示关注user的好友的隐向量
                        social_term_m += follower[1]*(p - ug) # 用在GD

                # 注释s rmse=0.83650
                # 不注释s rmse= 0.83391

                # update latent vectors
                self.P[u] += self.config.lr * (
                        error * q - self.config.alpha * (social_term_p + social_term_m) - self.config.lambdaP * p)
                self.Q[i] += self.config.lr * (error * p - self.config.lambdaQ * q)

                self.loss += 0.5 * self.config.alpha * social_term_loss

            self.loss += 0.5 * self.config.lambdaP * (self.P * self.P).sum() + 0.5 * self.config.lambdaQ * (
                    self.Q * self.Q).sum()

            iteration += 1
            if self.isConverged(iteration):
                break


if __name__ == '__main__':
    # srg = CUNE()
    # srg.train_model(0)
    # coldrmse = srg.predict_model_cold_users()
    # print('cold start user rmse is :' + str(coldrmse))
    # srg.show_rmse()

    rmses = []
    maes = []
    cunemf = CUNE()
    # cunemf.init_model(0)
    # cunemf.generate_cu_net()
    # cunemf.deep_walk()
    # print(bmf.rg.trainSet_u[1])
    cunemf.config.k_fold_num = 1
    for i in range(cunemf.config.k_fold_num): # 训练cunemf.config.k_fold_num次，取结果的均值
        print('the %dth cross validation training' % i)
        cunemf.train_model(i)
        rmse, mae = cunemf.predict_model()
        rmses.append(rmse)
        maes.append(mae)
    rmse_avg = sum(rmses) / cunemf.config.k_fold_num
    mae_avg = sum(maes) / cunemf.config.k_fold_num
    print("the rmses are %s" % rmses)
    print("the maes are %s" % maes)
    print("the average of rmses is %s " % rmse_avg)
    print("the average of maes is %s " % mae_avg)
