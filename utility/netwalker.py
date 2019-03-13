#encoding:utf-8
import sys
sys.path.append("..")
from time import time
from six import iterkeys
import logging
import random
from random import shuffle
from collections import defaultdict
from prettyprinter import cpprint
from configx.configx import ConfigX
import numpy as np

logger = logging.getLogger("deepwalk")

class Graph(defaultdict):
    """docstring for Graph"""
    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        return self.keys()

    def adjacency_iter(self):
        return self.iteritems()

    def make_consistent(self):
        t0 = time()
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))
        t1 = time()
        logger.info('make_consistent: made consistent in {}s'.format(t1-t0))
        self.remove_self_loops()
        return self

    def remove_self_loops(self):
        removed = 0
        t0 = time()
        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1

        t1 = time()
        logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
        return self

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        """ Returns a truncated random walk.
        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
        """
        G = self
        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur]))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]

def load_edgelist_without_weight(file_name, undirected=True):
    G = Graph()
    weight_dic = defaultdict(dict)
    with open(file_name) as f:
        for l in f:
            x, y = l.strip().split()[:2]
            x = int(x)
            y = int(y)
            G[x].append(y)
            if undirected:
                G[y].append(x)
    G.make_consistent()
    return G

def load_edgelist_with_weight(file_name, undirected=True):
    G = Graph()
    weight_dic = defaultdict(dict)
    with open(file_name) as f:
        for l in f:
            x, y, w = l.strip().split()[:3]
            x = int(x)
            y = int(y)
            w = round(float(w))

            weight_dic[x][y] = w

            for i in range(w):
                G[x].append(y)
                if undirected:
                    G[y].append(x)

    G.make_consistent()
    return weight_dic, G

def build_deepwalk_corpus(G, num_paths, path_length, alpha=0, random_state=0):
    walks = []
    rand = random.Random(random_state)
    nodes = list(G.nodes())
    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
    return walks

def deepwalk_with_alpha(G, weight_dic, num_paths, path_length, alpha=0, random_state=0):
    walks = []
    rand = random.Random(random_state)
    # nodes = list(G.nodes())
    for cnt in range(num_paths):
        # prob walk with alpha
        node = rand.choice(list(G.keys()))
        path = [node]
        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    _node_list = []
                    _weight_list = []
                    for _nbr in weight_dic[cur].keys():
                        _node_list.append(_nbr)
                        _weight_list.append(weight_dic[cur][_nbr])
                    _ps = [float(_weight) / sum(_weight_list) for _weight in _weight_list]
                    if _node_list != []:
                        sel_node = roulette(_node_list, _ps)
                        path.append(sel_node)
                    else:
                        break
                else:
                    path.append(path[0])
            else:
                break

        # rand.shuffle(nodes)
        # for node in nodes:
            # # start of random walk
            # path = [node]
            # while len(path) < path_length:
            #     cur = path[-1]
            #     if len(G[cur]) > 0:
            #         _node_list = []
            #         _weight_list = []
            #         for _nbr in weight_dic[cur].keys():
            #             _node_list.append(_nbr)
            #             _weight_list.append(weight_dic[cur][_nbr])
            #         _ps = [float(_weight) / sum(_weight_list) for _weight in _weight_list]
            #         sel_node = roulette(_node_list, _ps)
            #         path.append(sel_node)
            #         # current_word = sel_node
            #     else:
            #         break
            # end of random walk
        walk = [str(node) for node in path]
        walks.append(walk)
    return walks

def deepwalk_without_alpha(G, weight_dic, num_paths, path_length, alpha=0, random_state=0):
    walks = []
    rand = random.Random(random_state)
    # nodes = list(G.nodes())
    for cnt in range(num_paths):
        # prob walk without alpha
        node = rand.choice(list(G.keys()))
        path = [node]
        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                _node_list = []
                _weight_list = []
                for _nbr in weight_dic[cur].keys():
                    _node_list.append(_nbr)
                    _weight_list.append(weight_dic[cur][_nbr])
                _ps = [float(_weight) / sum(_weight_list) for _weight in _weight_list]
                if _node_list != []:
                    sel_node = roulette(_node_list, _ps)
                    path.append(sel_node)
                else:
                    break
                # node = sel_node
            else:
                break

        walk = [str(node) for node in path]
        walks.append(walk)
    return walks

def deepwalk_without_alpha_for_ex(G, ex_weight_dic, im_weight_dic, num_paths, path_length, alpha=0, random_state=0):
    walks = []
    rand = random.Random(random_state)
    # nodes = list(G.nodes())
    for cnt in range(num_paths):
        # prob walk without alpha
        node = rand.choice(list(G.keys()))
        path = [node]
        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                _node_list = []
                _weight_list = []
                for _nbr in ex_weight_dic[cur].keys():
                    _node_list.append(_nbr)
                    if cur in im_weight_dic.keys():
                        if _nbr in im_weight_dic[cur].keys():
                            _weight_list.append(im_weight_dic[cur][_nbr])
                        else:
                            _weight_list.append(ex_weight_dic[cur][_nbr])
                    else:
                        _weight_list.append(ex_weight_dic[cur][_nbr])
                _ps = [float(_weight) / sum(_weight_list) for _weight in _weight_list]
                if _node_list != []:
                    sel_node = roulette(_node_list, _ps)
                    path.append(sel_node)
                else:
                    break
                # node = sel_node
            else:
                break

        walk = [str(node) for node in path]
        walks.append(walk)
    return walks

def roulette(_datas, _ps):
    return np.random.choice(_datas, p=_ps)

def save_walks(walks, result_path):
    with open(result_path,"w") as f:
        for walk in walks:
            f.writelines(' '.join(walk)+'\n')
    pass


if __name__ == '__main__':
    g = Graph()
    c = ConfigX()
    number_walks = 5
    path_length = 5

    G=load_edgelist(c.trust_path,undirected=True)
    walks=build_deepwalk_corpus(G,number_walks,path_length)
    # cpprint(walks)
    save_walks(walks,"../data/social_corpus.txt")

    print("Number of nodes: {}".format(len(G.nodes())))

    num_walks = len(G.nodes()) * number_walks

    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * path_length

    print("Data size (walks*length): {}".format(data_size))