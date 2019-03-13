#encoding:utf-8
import sys
sys.path.append("..")

import os
from collections import defaultdict
import numpy as np
from prettyprinter import cpprint

from mf import MF
from reader.trust import TrustGetter
from configx.configx import ConfigX
from utility import util
from utility import netwalker,node_embedding
from utility.tools import sigmoid
# from utility.cross_validation import split_5_folds
from utility.netwalker import Graph
from utility.node_embedding import VocabItem,Vocab,UnigramTable,train_process
from random import choice
from reader.rating import RatingGetter
class GEMF(MF):
    """ python implementation for GEMF """
    def __init__(self):
        super(GEMF, self).__init__()
        self.rg = RatingGetter()
        ex_file = 'yp_trust'
        self.explict_trust_path = '../data/net/' + ex_file + '.txt'

        weight = 0.5
        # file = '%s_weight_%s' % (self.config.dataset_name, weight)
        file = 'yp_CUnet_weight'
        self.implict_trust_path = '../data/net/' + file + '.txt'
        # file = '%s_CUnet_weight_nnn' % self.config.dataset_name
        # file = '%s_less_CUnet_weight' % self.config.dataset_name
        # self.implict_trust_path = '../data/' + file + '.txt'
        # self.implict_trust_path = '../data/yp_30_39_rating_im_net_new.txt'  # ft_3 & db_13 & ca_16 & yp_30_39 # & ca_23 & db_18

        ############## 1 ################
        # ex_file = '%s_filter_trust_new' % self.config.dataset_name
        # file = '%s_CUnet_weight_new' % self.config.dataset_name
        # self.implict_trust_path = '../data/' + file + '.txt'
        # self.explict_trust_path = '../data/' + ex_file + '.txt'
        ############## 2 ################
        # file = 'ft_3_rating_im_net'
        # file = 'ft_3_rating_im_net_new' # ft_3 & db_18 & ca_23 & yp_30_39 for new
        # self.implict_trust_path = '../data/' + file + '.txt'
        ############## 3 ################
        # weight = 0.3
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
        self.config.lambdaP = 0.03 #0.03
        self.config.lambdaQ = 0.01 #0.01
        self.config.lambdaB = 0.01 #0.01
        self.config.temp1 = 0.01
        self.config.temp2 = 0.01
        self.config.alpha = self.config.temp1
        self.config.beta = self.config.temp2
        self.config.factor = 10
        self.config.isEarlyStopping = True
        self.config.k_fold_num = 5

        # parameters for netwalker
        self.config.random_state = 0
        self.config.number_walks = 30 # the times of random walk 5
        self.config.path_length = 20 # the length of random walk 10
        self.config.restart_pro = 0.1 # the probability of restarts.
        self.config.undirected = True
        self.config.ex_walk_result_path = '../data/ge/' + ex_file + '_social_corpus_filter.txt'
        self.config.im_walk_result_path = '../data/ge/' + file + '_social_corpus_implict.txt'
        # parameters for graph embedding
        self.config.lambdaW = 1
        self.config.ex_table_path = '../data/ge/' + ex_file + '_table_filter.pkl'
        self.config.ex_model_out_path = '../data/ge/' + ex_file + '_result_filter.txt'
        self.config.im_table_path = '../data/ge/' + file + '_table_implict.pkl'
        self.config.im_model_out_path = '../data/ge/' + file + '_result_implict.txt'
        self.config.cbow = 0
        self.config.neg = 5
        self.config.w2v_lr = 0.01  # 0.01-0.81
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
        # self.visited = defaultdict(dict)

        self.ex_pos_loss_total = 0
        self.ex_neg_loss_total = 0
        self.im_pos_loss_total = 0
        self.im_neg_loss_total = 0

        # cpprint('k is %s' % self.config.near_num)
        cpprint('implict_trust_path is %s' % self.implict_trust_path)
        cpprint('explict_trust_path is %s' % self.explict_trust_path)
        cpprint('lr is %s' % self.config.lr)
        cpprint('neg is %s' % self.config.neg)
        cpprint('w2v_lr is %s' % self.config.w2v_lr)
        cpprint('win_size is %s' % self.config.win_size)
        cpprint('alpha is %s' % self.config.alpha)
        cpprint('beta is %s' % self.config.beta)
        cpprint('lamdbaP is %s' % self.config.lambdaP)
        cpprint('lambdaQ is %s' % self.config.lambdaQ)
        cpprint('number_walks is %s' % self.config.number_walks)
        cpprint('path_length is %s' % self.config.path_length)
        # cpprint('factor is %s' % self.config.factor)

        self.init_model()

    def init_model(self):
        super(GEMF, self).init_model()
        print('starting initialization...')
        #1、extract user corpus with user's social network - netwalker
        print('='*5+'extracting user corpus with users social network'+'='*5)
        # ex_G = netwalker.load_edgelist_without_weight(self.explict_trust_path, undirected=self.config.undirected) # 读取显式用户网络 trust.txt
        # im_G = netwalker.load_edgelist_without_weight(self.implict_trust_path, undirected=self.config.undirected)  # 读取隐式用户网络 CUnet
        #
        # ex_weight_dic, ex_G = netwalker.load_edgelist_with_weight(self.explict_trust_path, undirected=self.config.undirected)  # 读取显式用户网络 trust.txt
        # im_weight_dic, im_G = netwalker.load_edgelist_with_weight(self.implict_trust_path, undirected=self.config.undirected)  # 读取隐式用户网络 CUnet
        # self.ex_walks = netwalker.build_deepwalk_corpus(ex_G, self.config.number_walks, self.config.path_length, self.config.restart_pro, self.config.random_state)
        # self.im_walks = netwalker.build_deepwalk_corpus(im_G, self.config.number_walks, self.config.path_length, self.config.restart_pro, self.config.random_state)

        ##########################################################
        # ex_weight_dic, ex_G = netwalker.load_edgelist_with_weight(self.explict_trust_path, undirected=self.config.undirected)  # 读取显式用户网络 trust.txt
        # im_weight_dic, im_G = netwalker.load_edgelist_with_weight(self.implict_trust_path, undirected=self.config.undirected)  # 读取隐式用户网络 CUnet
        # self.ex_walks = netwalker.deepwalk_with_alpha(ex_G, ex_weight_dic, self.config.number_walks, self.config.path_length, self.config.restart_pro, self.config.random_state)
        # self.im_walks = netwalker.deepwalk_with_alpha(im_G, im_weight_dic, self.config.number_walks, self.config.path_length, self.config.restart_pro, self.config.random_state)

        ########################################
        ex_weight_dic, ex_G = netwalker.load_edgelist_with_weight(self.explict_trust_path, undirected=self.config.undirected)  # 读取显式用户网络 trust.txt
        im_weight_dic, im_G = netwalker.load_edgelist_with_weight(self.implict_trust_path, undirected=self.config.undirected)  # 读取隐式用户网络 CUnet
        # self.ex_walks = netwalker.deepwalk_without_alpha(ex_G, ex_weight_dic, self.config.number_walks,
        #                                               self.config.path_length, self.config.restart_pro, self.config.random_state)
        self.ex_walks = netwalker.deepwalk_without_alpha_for_ex(ex_G, ex_weight_dic, im_weight_dic, self.config.number_walks,
                                                      self.config.path_length, self.config.restart_pro, self.config.random_state)
        self.im_walks = netwalker.deepwalk_without_alpha(im_G, im_weight_dic, self.config.number_walks,
                                                      self.config.path_length, self.config.restart_pro, self.config.random_state)
        # shuffle the walks
        np.random.shuffle(self.ex_walks)
        np.random.shuffle(self.im_walks)
        # cpprint(walks)
        netwalker.save_walks(self.ex_walks, self.config.ex_walk_result_path)
        netwalker.save_walks(self.im_walks, self.config.im_walk_result_path)
        print('='*5 + 'generating inverted index...' + '='*5)
        self.inverted_index()
        # print(self.node_inverted_index)

        #2、initialize the w and w' in graph embedding
        print('='*5+'read social corpus'+'='*5)
        ex_fi = open(self.config.ex_walk_result_path, 'r') # training corpus
        im_fi = open(self.config.im_walk_result_path, 'r')  # training corpus
        self.ex_social_vocab = Vocab(ex_fi, self.config.min_count) # user node and their index
        self.im_social_vocab = Vocab(im_fi, self.config.min_count) # user node and their index


        #social 的用户是否都在ui矩阵中出现，若是子集比较好说，若非子集则需将该用户随机初始化
        print('='*5+'initialize network for word2vec'+'='*5)
        self.reset_index(self.rg, self.im_social_vocab)
        self.init_net()

        print('='*5+'generate the unigram table for word2vec'+'='*5)
        if not os.path.exists(self.config.ex_table_path): # if exists, continue
            self.ex_table = UnigramTable(self.ex_social_vocab)
            util.save_data(self.ex_table, self.config.ex_table_path)
        else:
            self.ex_table =  util.load_data(self.config.ex_table_path)
        if not os.path.exists(self.config.im_table_path):
            self.im_table = UnigramTable(self.im_social_vocab)
            util.save_data(self.im_table, self.config.im_table_path)
        else:
            self.im_table =  util.load_data(self.config.im_table_path)

    def inverted_index(self):
        self.ex_node_inverted_index = defaultdict(set)
        self.im_node_inverted_index = defaultdict(set)
        for index, line in enumerate(self.ex_walks):
                for node in line:
                    self.ex_node_inverted_index[node].add(index)
        for index, line in enumerate(self.im_walks):
                for node in line:
                    self.im_node_inverted_index[node].add(index)
        pass

    def reset_index(self, rg, vocab): # rg表示rating抽取出来训练集的user。统计多少个social用户没有在ui中出现,需要将他们初始化然后并到P中去
        print('the current user number in ui is ' + str(len(self.rg.user)))
        not_exists_ui = [] # 记录不在ui中的user list
        num = 0
        mapping = {'<bol>': -1, '<eol>': -2, '<unk>': -3}
        for user in vocab.vocab_hash:
            if user != '<bol>' and user != '<eol>' and user != '<unk>' and not int(user) in rg.user:
                num += 1
                not_exists_ui.append(int(user))
                # 若社交语料的user不在ui阵里面，则将这个user扩充到训练集中，更新其编号，并且赋值给社交user字典
                self.rg.user[int(user)] = len(self.rg.user) # 扩充self.rg.user
                self.im_social_vocab.vocab_hash[user] = self.rg.user[int(user)] # reset the index of users in social corpus in order to be common in ui
            elif user != '<bol>' and user != '<eol>' and user != '<unk>':
                # 若社交语料的user在ui阵里面，则将这个user在ui阵训练集中的编号赋值给社交user字典
                self.im_social_vocab.vocab_hash[user] = self.rg.user[int(user)] # reset the index of users in social corpus
            else:
                # 处理三个特殊字符的编号
                index = mapping[user]
                self.rg.user[index] = len(self.rg.user)
                self.im_social_vocab.vocab_hash[user] = self.rg.user[index]
        print("the number of not exists in ui is "+ str(num))


    def init_net(self):
        # a = np.sqrt((self.config.max_val + self.config.min_val) / self.config.factor) 
        self.P = np.random.rand(self.rg.get_train_size()[0], self.config.factor) / (   #跟随机初始化的变量有很大关系
        self.config.factor ** 0.5)     # the common user latent vetors in MF and GE
        self.ex_W = np.random.rand(self.rg.get_train_size()[0], self.config.factor) / (
                self.config.factor ** 0.5)
        self.im_W = np.random.rand(self.rg.get_train_size()[0], self.config.factor) / (
                self.config.factor ** 0.5)
        self.Bu = np.random.rand(self.rg.get_train_size()[0]) / (self.config.factor ** 0.5) # bias value of user
        self.Bi = np.random.rand(self.rg.get_train_size()[1]) / (self.config.factor ** 0.5) 
        print('the shape of P is ' + str(self.P.shape))
        self.ex_pos_neu1 = np.zeros(self.config.factor)
        self.ex_neg_neul = np.zeros(self.config.factor)
        self.im_pos_neu1 = np.zeros(self.config.factor)
        self.im_neg_neul = np.zeros(self.config.factor)
        pass

    def train_model(self):
        super(GEMF, self).train_model()
        iteration = 0
        while iteration < self.config.maxIter:
            self.loss = 0
            self.ex_pos_loss_total = 0
            self.ex_neg_loss_total = 0
            self.im_pos_loss_total = 0
            self.im_neg_loss_total = 0
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line
                u = self.rg.user[user]
                i = self.rg.item[item]
                error = rating - self.predict(user, item)
                self.loss += 0.5 * error ** 2 # 需要最后计算loss吧？
                p, q = self.P[u], self.Q[i]
                bu, bi = self.Bu[u], self.Bi[i]

                # the formulation of w2v
                if str(user) not in self.ex_node_inverted_index:
                    self.im_pos_loss_total += 0
                    self.im_neg_loss_total += 0
                    self.config.alpha = 0
                else:
                    self.config.alpha = self.config.temp1
                    im_walks_list = self.im_node_inverted_index[str(user)]
                    im_wl = len(im_walks_list)

                    if im_wl > 0:
                        rand_num = np.random.randint(low=0, high=len(im_walks_list))
                        # print(rand_num)
                        im_line_num = list(im_walks_list)[rand_num]
                        im_walks = self.im_walks[im_line_num]
                        im_sent = self.im_social_vocab.indices(['<bol>'] + im_walks + ['<eol>'])
                        for im_sent_pos, im_token in enumerate(im_sent):
                            if im_token != u:
                                continue
                            im_current_win = self.config.win_size  # np.random.randint(low=1, high=self.config.win_size+1)
                            im_context_start = max(im_sent_pos - im_current_win, 0)
                            im_context_end = min(im_sent_pos + im_current_win + 1, len(im_sent))
                            im_context = im_sent[im_context_start:im_sent_pos] + im_sent[im_sent_pos + 1:im_context_end]  # 取出中心词的上下文context

                            # neu1 = np.mean(np.array([self.W[c] for c in self.context]), axis=0) # 压缩行，对各列求均值，计算上下文行均值得到 h of N+

                            self.im_pos_neu1 = np.mean(np.array([self.im_W[c] for c in im_context]), axis=0)
                            self.im_neg_neul = np.mean(np.array([self.im_W[target] for target in self.im_table.sample(self.config.neg)]), axis=0)
                            self.im_pos_loss_total += np.log(sigmoid(np.dot(self.P[im_token], self.im_pos_neu1)))
                            self.im_neg_loss_total += np.log(sigmoid(- np.dot(self.P[im_token], self.im_neg_neul)))
                            if self.config.neg > 0:
                                # classifiers = [(token, 1)] + [(target, 0) for target in self.table.sample(self.config.neg)]
                                im_classifiers = [(im_context_word, 1) for im_context_word in im_context] + [(im_target, 0) for im_target in
                                                                                                 self.im_table.sample(self.config.neg)]
                            for im_word, im_label in im_classifiers:
                                if im_label == 1:
                                    z_po = np.dot(self.im_pos_neu1, self.P[im_token])
                                    p_po = sigmoid(z_po)
                                    f_po = self.config.beta * (im_label - p_po)
                                    g_po = f_po * self.P[im_token] - self.config.lambdaW * self.im_W[im_word] - (self.im_W[im_word] - self.ex_W[im_word])  # 负梯度
                                    self.im_W[im_word] += self.config.w2v_lr * g_po
                                else:
                                    z_ne = np.dot(self.im_neg_neul, self.P[im_token])
                                    p_ne = sigmoid(z_ne)
                                    f_ne = self.config.beta * (im_label - p_ne)
                                    g_ne = f_ne * self.P[im_token] - self.config.lambdaW * self.im_W[im_word] - (self.im_W[im_word] - self.ex_W[im_word])  # 负梯度
                                    self.im_W[im_word] += self.config.w2v_lr * g_ne
                                
                if str(user) not in self.ex_node_inverted_index:
                    self.ex_pos_loss_total += 0
                    self.ex_neg_loss_total += 0
                    self.config.beta = 0
                else:
                    self.config.beta = self.config.temp2
                    ex_walks_list = self.ex_node_inverted_index[str(user)]
                    ex_wl = len(ex_walks_list)

                    if ex_wl > 0:
                        # for walk_line in walks_list:
                        # np.random.seed(10)
                        rand_num = np.random.randint(low=0, high=len(ex_walks_list))
                        # print(rand_num)
                        ex_line_num = list(ex_walks_list)[rand_num]
                        ex_walks = self.ex_walks[ex_line_num]
                        ex_sent = self.ex_social_vocab.indices(['<bol>'] + ex_walks + ['<eol>'])
                        # self.pos_total = 0
                        # self.neg_total = 0
                        # self.context = {}
                        # self.pos_loss_total = 0
                        # self.neg_loss_total = 0
                        for ex_sent_pos, ex_token in enumerate(ex_sent):
                            if ex_token != u:
                                continue
                            ex_current_win = self.config.win_size  # np.random.randint(low=1, high=self.config.win_size+1)
                            ex_context_start = max(ex_sent_pos - ex_current_win, 0)
                            ex_context_end = min(ex_sent_pos + ex_current_win + 1, len(ex_sent))
                            ex_context = ex_sent[ex_context_start:ex_sent_pos] + ex_sent[ex_sent_pos + 1:ex_context_end]  # 取出中心词的上下文context

                            # neu1 = np.mean(np.array([self.W[c] for c in self.context]), axis=0) # 压缩行，对各列求均值，计算上下文行均值得到 h of N+

                            self.ex_pos_neu1 = np.mean(np.array([self.ex_W[c] for c in ex_context]), axis=0)
                            self.ex_neg_neul = np.mean(np.array([self.ex_W[target] for target in self.ex_table.sample(self.config.neg)]), axis=0)
                            self.ex_pos_loss_total += np.log(sigmoid(np.dot(self.P[ex_token], self.ex_pos_neu1)))
                            self.ex_neg_loss_total += np.log(sigmoid(- np.dot(self.P[ex_token], self.ex_neg_neul)))
                            if self.config.neg > 0:
                                # classifiers = [(token, 1)] + [(target, 0) for target in self.table.sample(self.config.neg)]
                                ex_classifiers = [(ex_context_word, 1) for ex_context_word in ex_context] \
                                                 + [(ex_target, 0) for ex_target in self.ex_table.sample(self.config.neg)]
                            for ex_word, ex_label in ex_classifiers:
                                if ex_label == 1:
                                    z_po = np.dot(self.ex_pos_neu1, self.P[ex_token])
                                    p_po = sigmoid(z_po)
                                    f_po = self.config.alpha * (ex_label - p_po)
                                    g_po = f_po * self.P[ex_token] - self.config.lambdaW * self.ex_W[ex_word] + (self.im_W[ex_word] - self.ex_W[ex_word])  # 负梯度
                                    self.ex_W[ex_word] += self.config.w2v_lr * g_po
                                else:
                                    z_ne = np.dot(self.ex_neg_neul, self.P[ex_token])
                                    p_ne = sigmoid(z_ne)
                                    f_ne = self.config.alpha * (ex_label - p_ne)
                                    g_ne = f_ne * self.P[ex_token] - self.config.lambdaW * self.ex_W[ex_word] + (self.im_W[ex_word] - self.ex_W[ex_word])  # 负梯度
                                    self.ex_W[ex_word] += self.config.w2v_lr * g_ne

                # update latent vectors P, Q, Bu, Bi in MF
                self.Bu[u] += self.config.lr * (error - self.config.lambdaB * bu)
                self.Bi[i] += self.config.lr * (error - self.config.lambdaB * bi)
                self.P[u] += self.config.lr * (error * q
                        + self.config.alpha * (self.ex_pos_neu1 * (1 - sigmoid(np.dot(self.ex_pos_neu1, self.P[u])))
                        - self.ex_neg_neul * sigmoid(np.dot(self.ex_neg_neul, self.P[u])))
                        + self.config.beta * (self.im_pos_neu1 * (1 - sigmoid(np.dot(self.im_pos_neu1, self.P[u])))
                        - self.im_neg_neul * sigmoid(np.dot(self.im_neg_neul, self.P[u]))) - self.config.lambdaP * p)
                self.Q[i] += self.config.lr * (error * p - self.config.lambdaQ * q)

            # self.loss += 0.5 * self.config.lambdaP * (self.P * self.P).sum() + 0.5 * self.config.lambdaQ * (
            #         self.Q * self.Q).sum() + 0.5 * self.config.lambdaB * ((self.Bu * self.Bu).sum() + (self.Bi * self.Bi).sum())
            self.loss += self.config.alpha * (- self.ex_pos_loss_total - self.ex_neg_loss_total) \
                         + self.config.beta * (- self.im_pos_loss_total - self.im_neg_loss_total) \
                         + 0.5 * self.config.lambdaP * (self.P * self.P).sum() \
                         + 0.5 * self.config.lambdaQ * (self.Q * self.Q).sum() \
                         + 0.5 * self.config.lambdaB * ((self.Bu * self.Bu).sum() + (self.Bi * self.Bi).sum()) \
                         + 0.5 * self.config.lambdaW * ((self.ex_W * self.ex_W).sum() + (self.im_W * self.im_W).sum())
                         # + 0.5 * np.linalg.norm(self.im_W - self.ex_W) ** 2
            iteration += 1
            if self.isConverged(iteration):
                break

    # def predict(self, u, i):
    #     if self.rg.containsUser(u) and self.rg.containsItem(i):
    #         u = self.rg.user[u]
    #         i = self.rg.item[i]
    #         return self.P[u].dot(self.Q[i]) + self.rg.globalMean + self.Bi[i] + self.Bu[u]
    #     else:
    #         return self.rg.globalMean

    # def predict(self, u, i):
    #     if self.rg.containsUser(u) and self.rg.containsItem(i):
    #         u = self.rg.user[u]
    #         i = self.rg.item[i]
    #         return self.P[u].dot(self.Q[i]) + self.rg.globalMean + self.Bi[i] + self.Bu[u]
    #     elif self.rg.containsUser(u) and not self.rg.containsItem(i):
    #         return self.rg.userMeans[u]
    #     elif not self.rg.containsUser(u) and self.rg.containsItem(i):
    #         return self.rg.itemMeans[i]
    #     else:
    #         return self.rg.globalMean

    def generate_cu_net(self, rg):
        f = open(self.implict_trust_path,'w')
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
                        # self.CUNet[user1] += [user2] # 得到{user1:[user2,......]}
                        f.writelines(str(user1) + ' ' + str(user2) + ' ' + str(weight) + '\n')

if __name__ == '__main__':

    #运行一次就行
    # configx = ConfigX()
    # configx.k_fold_num = 5 
    # configx.rating_path = "../data/" + configx.dataset_name + "_ratings.txt"
    # configx.rating_cv_path = "../data/cv/"
    
    # split_5_folds(configx)

    # gemf = GEMF()
    # gemf.generate_cu_net(gemf.rg)

    rmses = []
    maes = []
    cold_rmses = []
    cold_maes = []
    gemf = GEMF()
    gemf.train_model()
    rmse, mae = gemf.predict_model()
