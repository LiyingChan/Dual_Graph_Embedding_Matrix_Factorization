#encoding:utf-8
import sys
sys.path.append("..")

import os
from collections import defaultdict
import numpy as np
from prettyprinter import cpprint

from mf import MF
# from reader.trust_ex import TrustGetter
from configx.configx import ConfigX
from utility import util
from utility import netwalker, node_embedding
from utility.tools import sigmoid
from utility.node_embedding import VocabItem,Vocab,UnigramTable,train_process
from random import choice
class GEMF(MF):
    """ python implementation for GEMF """
    def __init__(self):
        super(GEMF, self).__init__()
        # file = '%s_filter_trust' % self.config.dataset_name
        weight = 0.5
        file = '%s_CUnet' % self.config.dataset_name
        # file = '%s_tanh' % (self.config.dataset_name)
        self.implict_trust_path = '../data/net/' + file + '.txt'
        # self.implict_trust_path = '../data/yp_30_39_rating_im_net_new.txt'  # ft_3 & db_13 & ca_16 & yp_30_39 # & ca_23 & db_18

        ############## 1 ################
        # file = '%s_filter_trust_new' % self.config.dataset_name
        # file = '%s_CUnet_weight_new' % self.config.dataset_name
        # self.implict_trust_path = '../data/' + file + '.txt'
        ############## 2 ################
        # file = 'ft_3_rating_im_net'
        # file = 'ft_3_rating_im_net_new' # ft_3 & db_18 & ca_23 & yp_30_39 for new
        # self.implict_trust_path = '../data/' + file + '.txt'
        ############## 3 ################
        # weight = 0.9
        # file = '%s_two_net_with_weight_%s_rewrited' % (self.config.dataset_name, weight)
        # file = '%s_two_net_with_weight_%s_new_rewrited' % (self.config.dataset_name, weight)
        # self.implict_trust_path = '../data/%s_two_net/' % self.config.dataset_name + file + '.txt'
        ############## 4 ################
        # file = '%s_two_net_with_tanh_rewrited' % (self.config.dataset_name)
        # file = '%s_two_net_with_tanh_new_rewrited' % (self.config.dataset_name)
        # self.implict_trust_path = '../data/%s_two_net/' % self.config.dataset_name + file + '.txt'
        ############## 5 ################
        # file = '%s_inter_net' % self.config.dataset_name
        # file = '%s_union_net' % self.config.dataset_name
        # file = '%s_union_net_expanded' % self.config.dataset_name
        # file = '%s_inter_net_new' % self.config.dataset_name
        # file = '%s_union_net_new' % self.config.dataset_name
        # file = '%s_union_net_new_expanded' % self.config.dataset_name
        # self.implict_trust_path = '../data/%s_two_net/' % self.config.dataset_name + file + '.txt'

        # parameters for matrix factorization
        self.config.lr = 0.01
        self.config.lambdaP = 0.3 #0.03
        self.config.lambdaQ = 0.3 #0.01
        self.config.lambdaB = 0.3 #0.01
        self.config.alpha = 0.01
        # self.config.beta = 0.02
        self.config.factor = 40
        self.config.isEarlyStopping = True
        self.config.k_fold_num = 5

        # parameters for netwalker
        self.config.random_state = 0
        self.config.number_walks = 30 # the times of random walk 5
        self.config.path_length = 100 # the length of random walk 10
        self.config.restart_pro = 0.1 # the probability of restarts.
        self.config.undirected = True
        self.config.walk_result_path = '../data/ge/' + str(self.config.path_length) + '_social_corpus_im_only.txt'

        # parameters for graph embedding
        self.config.lambdaW = 1
        self.config.table_path = '../data/ge/' + str(self.config.path_length) + '_table_im_only.pkl'
        self.config.model_out_path = '../data/ge/' + str(self.config.path_length) + '_result_im_only.txt'
        self.config.cbow = 0
        self.config.neg = 5
        self.config.w2v_lr = 0.01 # 0.01-0.81
        self.config.win_size = 10
        self.config.min_count = 3
        self.config.binary = 0

        self.dataSet_u = defaultdict(dict)
        self.dataSet_i = defaultdict(dict)
        self.filteredRatings = defaultdict(list)
        self.CUNet = defaultdict(list)
        self.walks = []
        self.ex_walks = []
        self.im_walks = []
        self.visited = defaultdict(dict)

        # self.pos_total = 0
        # self.neg_total = 0
        # self.context = {}
        self.pos_loss_total = 0
        self.neg_loss_total = 0

        cpprint('path is %s' % self.implict_trust_path)
        cpprint('weight is %s' % weight)
        cpprint('lr is %s' % self.config.lr)
        cpprint('w2v_lr is %s' % self.config.w2v_lr)
        cpprint('neg is %s' % self.config.neg)
        cpprint('win_size is %s' % self.config.win_size)
        # cpprint('k is %s' % self.near_num)

    def init_model(self):
        super(GEMF, self).init_model()
        print('starting initialization...')
        #1、extract user corpus with user's social network - netwalker
        print('='*5+'extracting user corpus with users social network'+'='*5)
        ##########################
        # G = netwalker.load_edgelist_without_weight(self.implict_trust_path, undirected=self.config.undirected)
        # self.walks = netwalker.build_deepwalk_corpus(G, self.config.number_walks, self.config.path_length, self.config.restart_pro, self.config.random_state)

        # weight_dic, G = netwalker.load_edgelist_with_weight(self.implict_trust_path, undirected=self.config.undirected)
        # self.walks = netwalker.build_deepwalk_corpus(G, self.config.number_walks, self.config.path_length, self.config.restart_pro, self.config.random_state)

        ##########################
        # weight_dic, G = netwalker.load_edgelist_with_weight(self.implict_trust_path, undirected=self.config.undirected)
        # self.walks = netwalker.deepwalk_with_alpha(G, weight_dic, self.config.number_walks, self.config.path_length, self.config.restart_pro, self.config.random_state)

        weight_dic, G = netwalker.load_edgelist_with_weight(self.implict_trust_path, undirected=self.config.undirected)
        self.walks = netwalker.deepwalk_without_alpha(G, weight_dic, self.config.number_walks, self.config.path_length, self.config.restart_pro, self.config.random_state)

        # # shuffle the walks
        np.random.shuffle(self.walks)
        # cpprint(walks)
        netwalker.save_walks(self.walks, self.config.walk_result_path)
        print('='*5 + 'generating inverted index...' + '='*5)
        self.inverted_index()
        # print(self.node_inverted_index)

        #2、initialize the w and w' in graph embedding
        print('='*5+'read social corpus'+'='*5)
        fi = open(self.config.walk_result_path, 'r') # training corpus
        self.social_vocab = Vocab(fi, self.config.min_count) # user node and their index


        #social 的用户是否都在ui矩阵中出现，若是子集比较好说，若非子集则需将该用户随机初始化
        print('='*5+'initialize network for word2vec'+'='*5)
        self.reset_index()
        self.init_net()

        print('='*5+'generate the unigram table for word2vec'+'='*5)
        if not os.path.exists(self.config.table_path): # if exists, continue
            self.table = UnigramTable(self.social_vocab)
            util.save_data(self.table, self.config.table_path)
        else:
            self.table =  util.load_data(self.config.table_path)

    def inverted_index(self):
        self.node_inverted_index = defaultdict(set)
        for index, line in enumerate(self.walks):
                for node in line:
                    self.node_inverted_index[node].add(index)
        pass

    def reset_index(self): # 141个social用户没有在ui中出现,需要将他们初始化然后并到P中去
        print('the current user number in ui is ' + str(len(self.rg.user)))
        not_exists_ui = [] # 记录不在ui中的user list
        num = 0
        mapping = {'<bol>': -1, '<eol>': -2, '<unk>': -3}
        for user in self.social_vocab.vocab_hash:
            if user != '<bol>' and user != '<eol>' and user != '<unk>' and not int(user) in self.rg.user:
                num += 1
                not_exists_ui.append(int(user))
                # 若社交语料的user不在ui阵里面，则将这个user扩充到训练集中，更新其编号，并且赋值给社交user字典
                self.rg.user[int(user)] = len(self.rg.user) # 扩充self.rg.user
                self.social_vocab.vocab_hash[user] = self.rg.user[int(user)] # reset the index of users in social corpus in order to be common in ui
            elif user != '<bol>' and user != '<eol>' and user != '<unk>':
                # 若社交语料的user在ui阵里面，则将这个user在ui阵训练集中的编号赋值给社交user字典
                self.social_vocab.vocab_hash[user] = self.rg.user[int(user)] # reset the index of users in social corpus
            else:
                # 处理三个特殊字符的编号
                index = mapping[user]
                self.rg.user[index] = len(self.rg.user)
                self.social_vocab.vocab_hash[user] = self.rg.user[index]
        print("the number of not exists in ui is "+ str(num))


    def init_net(self):
        # a = np.sqrt((self.config.max_val + self.config.min_val) / self.config.factor)
        self.P = np.random.rand(self.rg.get_train_size()[0], self.config.factor) / (   #跟随机初始化的变量有很大关系
        self.config.factor ** 0.5)     # the common user latent vetors in MF and GE
        self.Q = np.random.rand(self.rg.get_train_size()[1], self.config.factor) / (  # 跟随机初始化的变量有很大关系
                self.config.factor ** 0.5)  # the common user latent vetors in MF and GE
        self.W = np.random.rand(self.rg.get_train_size()[0], self.config.factor) / (
        self.config.factor ** 0.5)   # the common user latent vetors in MF and GE
        self.Bu = np.random.rand(self.rg.get_train_size()[0]) / (self.config.factor ** 0.5) # bias value of user
        self.Bi = np.random.rand(self.rg.get_train_size()[1]) / (self.config.factor ** 0.5)
        print('the shape of P is ' + str(self.P.shape))
        self.pos_neu1 = np.zeros(self.config.factor)
        self.neg_neul = np.zeros(self.config.factor)

        pass

    def train_model(self):
        super(GEMF, self).train_model()
        iteration = 0
        while iteration < self.config.maxIter:
            self.loss = 0
            self.pos_loss_total = 0
            self.neg_loss_total = 0
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line
                u = self.rg.user[user]
                i = self.rg.item[item]
                error = rating - self.predict(user, item)
                self.loss += 0.5 * error ** 2 # 需要最后计算loss吧？
                p, q = self.P[u], self.Q[i]
                bu, bi = self.Bu[u], self.Bi[i]

                # the formulation of w2v
                walks_list = self.node_inverted_index[str(user)]
                wl = len(walks_list)
                neu1e = np.zeros(self.config.factor)

                if wl > 0:
                # for walk_line in walks_list:
                    # np.random.seed(10)
                    rand_num = np.random.randint(low=0, high=len(walks_list))
                    # print(rand_num)
                    line_num = list(walks_list)[rand_num]
                    walks = self.walks[line_num]
                    sent = self.social_vocab.indices(['<bol>'] + walks + ['<eol>'])
                    # self.pos_total = 0
                    # self.neg_total = 0
                    # self.context = {}
                    # self.pos_loss_total = 0
                    # self.neg_loss_total = 0
                    for sent_pos, token in enumerate(sent):
                        if token != u:
                            continue
                        current_win = self.config.win_size #np.random.randint(low=1, high=self.config.win_size+1)
                        context_start = max(sent_pos - current_win, 0)
                        context_end = min(sent_pos + current_win + 1, len(sent))
                        context = sent[context_start:sent_pos] + sent[sent_pos+1:context_end] # 取出中心词的上下文context

                        # neu1 = np.mean(np.array([self.W[c] for c in self.context]), axis=0) # 压缩行，对各列求均值，计算上下文行均值得到 h of N+

                        self.pos_neu1 = np.mean(np.array([self.W[c] for c in context]), axis=0)
                        self.neg_neul = np.mean(np.array([self.W[target] for target in self.table.sample(self.config.neg)]), axis=0)
                        self.pos_loss_total += np.log(sigmoid(np.dot(self.P[token], self.pos_neu1)))
                        self.neg_loss_total += np.log(sigmoid(- np.dot(self.P[token], self.neg_neul)))
                        if self.config.neg > 0:
                            # classifiers = [(token, 1)] + [(target, 0) for target in self.table.sample(self.config.neg)]
                            classifiers = [(context_word, 1) for context_word in context] + [(target, 0) for target in self.table.sample(self.config.neg)]
                        for target, label in classifiers:
                            if label == 1:
                                # z = np.dot(self.pos_neu1, self.P[target])
                                # p = sigmoid(z)
                                # g = self.config.w2v_lr * (label - p) # 负梯度
                                # self.pos_total += g * self.P[target]
                                z_po = np.dot(self.pos_neu1, self.P[token])
                                p_po = sigmoid(z_po)
                                f_po = self.config.alpha * (label - p_po)
                                g_po = f_po * self.P[token] - self.config.lambdaW * self.W[target]  # 负梯度
                                self.W[target] += self.config.w2v_lr * g_po
                            else:
                                z_ne = np.dot(self.neg_neul, self.P[token])
                                p_ne = sigmoid(z_ne)
                                f_ne = self.config.alpha * (label - p_ne)
                                g_ne = f_ne * self.P[token] - self.config.lambdaW * self.W[target]  # 负梯度
                                self.W[target] += self.config.w2v_lr * g_ne

                        # 更新W
                        # for context_word in context:
                        #     self.W[context_word] += self.config.lambdaW * neu1e
                        # for context_word in self.context[u]:
                        #     # s = len(self.context[u])
                        #     self.W[context_word] += self.config.alpha * self.pos_total - self.config.lambdaW * self.W[context_word]
                        # for target in self.table.sample(self.config.neg):
                        #     self.W[target] += self.config.alpha * self.neg_total - self.config.lambdaW * self.W[target]
                        # self.pos_neu1 = np.mean(np.array([self.W[c] for c in self.context[token]]), axis=0)
                        # self.neg_neul = np.mean(np.array([self.W[target] for target in self.table.sample(self.config.neg)]), axis=0)
                        # self.pos_total += (1 - sigmoid(np.dot(self.P[token], self.pos_neu1))) * self.P[token]
                        # self.neg_total += (- sigmoid(np.dot(self.P[token], self.neg_neul)) * self.P[token])

                    # for context_word in self.context[u]:
                    #     s = len(self.context[u])
                    #     self.W[context_word] += (- self.config.alpha * self.pos_total / s - self.config.lambdaW * self.W[context_word])
                    # for target in self.table.sample(self.config.neg):
                    #     self.W[target] += (- self.config.alpha * self.neg_total / self.config.neg - self.config.lambdaW * self.W[target])
                # update latent vectors P, Q, Bu, Bi in MF
                self.Bu[u] += self.config.lr * (error - self.config.lambdaB * bu)
                self.Bi[i] += self.config.lr * (error - self.config.lambdaB * bi)
                self.P[u] += self.config.lr * (
                        error * q + self.config.alpha * (self.pos_neu1 * (1 - sigmoid(np.dot(self.pos_neu1, self.P[u])))
                                                         - self.neg_neul * sigmoid(np.dot(self.neg_neul, self.P[u]))) - self.config.lambdaP * p)
                self.Q[i] += self.config.lr * (error * p - self.config.lambdaQ * q)

                # self.pos_loss_total += np.log(sigmoid(np.dot(self.P[u], self.pos_neu1)))
                # self.neg_loss_total += np.log(sigmoid(- np.dot(self.P[u], self.neg_neul)))
                # self.loss += self.config.alpha * (- self.pos_loss_total - self.neg_loss_total)
            self.loss += self.config.alpha * (- self.pos_loss_total - self.neg_loss_total) + 0.5 * self.config.lambdaP * (self.P * self.P).sum() + 0.5 * self.config.lambdaQ * (
                    self.Q * self.Q).sum() + 0.5 * self.config.lambdaB * ((self.Bu * self.Bu).sum() + (self.Bi * self.Bi).sum()) + 0.5 * self.config.lambdaW * (self.W * self.W).sum()
            iteration += 1
            if self.isConverged(iteration):
                break

if __name__ == '__main__':

    #运行一次就行
    # configx = ConfigX()
    # configx.k_fold_num = 5
    # configx.rating_path = "../data/" + configx.dataset_name + "_ratings.txt"
    # configx.rating_cv_path = "../data/cv/"

    # split_5_folds(configx)

    gemf = GEMF()
    gemf.init_model()
    gemf.train_model()
    rmse, mae = gemf.predict_model()
    # cold_rmse,cold_mae = gemf.predict_model_cold_users()


    print("test rmses are %s" % rmse)
    print("test maes are %s" % mae)

    # cpprint(gemf.config.__dict__)
    # print("the cold rmses are %s" % cold_rmses)
    # print("the cold maes are %s" % cold_maes)
    # print("the average of cold rmses is %s " % cold_rmse_avg)
    # print("the average of cold maes is %s " % cold_mae_avg)
