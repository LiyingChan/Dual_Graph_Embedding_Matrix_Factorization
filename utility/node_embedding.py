#encoding:utf-8
import sys
sys.path.append("..") #将该目录加入到环境变量

import argparse
import math
import struct
import sys
import time
import warnings
import os

import numpy as np

#每个单词的结构体
class VocabItem:
    def __init__(self, word):
        self.word = word
        self.count = 0

#整个语料库的词汇
class Vocab:
    def __init__(self, fi, min_count):
        vocab_items = []
        vocab_hash = {}
        word_count = 0

        # Add special tokens <bol> (beginning of line) and <eol> (end of line)
        for token in ['<bol>', '<eol>']:
            vocab_hash[token] = len(vocab_items)
            vocab_items.append(VocabItem(token))

        for line in fi:
            tokens = line.split() #默认为空格
            for token in tokens:
                if token not in vocab_hash:
                    vocab_hash[token] = len(vocab_items)
                    vocab_items.append(VocabItem(token))
                    
                # assert vocab_items[vocab_hash[token]].word == token, 'Wrong vocab_hash index'
                vocab_items[vocab_hash[token]].count += 1
                word_count += 1
            
                # if word_count % 10000 == 0:
                #     sys.stdout.write("\rReading word %d\n" % word_count)
                #     sys.stdout.flush()

            # Add special tokens <bol> (beginning of line) and <eol> (end of line)
            vocab_items[vocab_hash['<bol>']].count += 1
            vocab_items[vocab_hash['<eol>']].count += 1
            word_count += 2

        self.bytes = fi.tell()
        self.vocab_items = vocab_items         # List of VocabItem objects
        self.vocab_hash = vocab_hash           # Mapping from each token to its index in vocab
        self.word_count = word_count           # Total number of words in train file

        # Add special token <unk> (unknown),
        # merge words occurring less than min_count into <unk>, and
        # sort vocab in descending order by frequency in train file
        self.__sort(min_count)

        # assert self.word_count == sum([t.count for t in self.vocab_items]), 'word_count and sum of t.count do not agree'
        # print('Total words in training file: %d' % self.word_count)
        # print('Total bytes in training file: %d' % self.bytes)
        print('Vocab size in social network: %d' % len(self))

    def __getitem__(self, i):
        return self.vocab_items[i]

    def __len__(self):
        return len(self.vocab_items)

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, key):
        return key in self.vocab_hash

    def __sort(self, min_count):
        tmp = []
        tmp.append(VocabItem('<unk>'))
        unk_hash = 0
        
        count_unk = 0
        for token in self.vocab_items:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token : token.count, reverse=True)

        # Update vocab_hash
        vocab_hash = {}
        for i, token in enumerate(tmp):
            vocab_hash[token.word] = i

        self.vocab_items = tmp
        self.vocab_hash = vocab_hash

        # print('Unknown vocab size:', count_unk)

    def indices(self, tokens):
        return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens]


class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """
    def __init__(self, vocab):
        vocab_size = len(vocab)
        power = 0.75 #0.75
        norm = sum([math.pow(t.count, power) for t in vocab]) # Normalizing constant

        table_size = 1e8 #1e8 # Length of the unigram table
        table = np.zeros(int(table_size), dtype=np.uint32)

        print('Filling unigram table')
        p = 0 # Cumulative probability
        i = 0
        for j, unigram in enumerate(vocab):
            # print(j)
            # print(unigram)
            p += float(math.pow(unigram.count, power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table
        # table_path='../data/ge/table.txt'
        # save_table(table,table_path)

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]

def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))


def init_net(dim, vocab_size):
    # Init syn0 with random numbers from a uniform distribution on the interval [-0.5, 0.5]/dim
    tmp = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(vocab_size, dim))
    # syn0 = np.ctypeslib.as_ctypes(tmp)
    # syn0 = Array(syn0._type_, syn0, lock=False)
    syn0 = tmp

    # Init syn1 with zeros
    tmp = np.zeros(shape=(vocab_size, dim))
    # syn1 = np.ctypeslib.as_ctypes(tmp)
    # syn1 = Array(syn1._type_, syn1, lock=False)
    syn1 = tmp

    return (syn0, syn1)


def train_process(fi, social_vocab, P, W, table, fo, cbow, neg, dim, alpha, win, min_count, binary):
    print('graph embedding training...')
    fi=open(fi,'r')
    vocab = social_vocab
    # Init net
    syn0, syn1 = W,P

    # global_word_count = Value('i', 0)
    # table = None
   
    # table = UnigramTable(vocab)

    t0 = time.time()
    

    word_count = 0
    last_word_count = 0

    for index, line in enumerate(fi):
        line = line.strip()
        # if index%500==0:
        #     print('processing %d th line'%index)
        # Skip blank lines
        if not line:
            continue

        # Init sent, a list of indices of words in line
        sent = vocab.indices(['<bol>'] + line.split() + ['<eol>'])

        for sent_pos, token in enumerate(sent):

            # Randomize window size, where win is the max window size
            current_win = np.random.randint(low=1, high=win+1)
            context_start = max(sent_pos - current_win, 0)
            context_end = min(sent_pos + current_win + 1, len(sent))
            context = sent[context_start:sent_pos] + sent[sent_pos+1:context_end] # Turn into an iterator?
            # CBOW
            if cbow:
                # Compute neu1
                neu1 = np.mean(np.array([syn0[c] for c in context]), axis=0)
                assert len(neu1) == dim, 'neu1 and dim do not agree'

                # Init neu1e with zeros
                neu1e = np.zeros(dim)

                # Compute neu1e and update syn1
                if neg > 0:
                    classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                else:
                    classifiers = zip(vocab[token].path, vocab[token].code)
                for target, label in classifiers:
                    z = np.dot(neu1, syn1[target])
                    p = sigmoid(z)
                    g = alpha * (label - p)
                    neu1e += g * syn1[target] # Error to backpropagate to syn0
                    syn1[target] += g * neu1  # Update syn1

                # Update syn0
                for context_word in context:
                    syn0[context_word] += neu1e

            # Skip-gram
            else:
                for context_word in context:
                    # Init neu1e with zeros
                    neu1e = np.zeros(dim)

                    # Compute neu1e and update syn1
                    if neg > 0:
                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                    else:
                        classifiers = zip(vocab[token].path, vocab[token].code)
                    for target, label in classifiers:
                        z = np.dot(syn0[context_word], syn1[target])
                        p = sigmoid(z)
                        g = alpha * (label - p)
                        neu1e += g * syn1[target]              # Error to backpropagate to syn0
                        syn1[target] += 0.001 * g * syn0[context_word] # Update syn1

                    # Update syn0
                    syn0[context_word] += 0.001 * neu1e

            word_count += 1

    fi.close()

    t1 = time.time()
    print('Completed training. Training took', (t1 - t0) / 60, 'minutes')
    return syn0, syn1
    # save_model(vocab, syn0, fo, binary)

def save_table(table,table_path):
    print('Saving table to',table_path)
    fo=open(table_path,'w')
    for item in table:
        fo.write(str(item)+' '+str(table[item])+'\n')
    fo.close()

def save_model(vocab, syn0, fo, binary):
    print('Saving model to', fo)
    dim = len(syn0[0])
    if binary:
        fo = open(fo, 'wb')
        fo.write('%d %d\n' % (len(syn0), dim))
        fo.write('\n')
        for token, vector in zip(vocab, syn0):
            fo.write('%s ' % token.word)
            for s in vector:
                fo.write(struct.pack('f', s))
            fo.write('\n')
    else:
        fo = open(fo, 'w')
        fo.write('%d %d\n' % (len(syn0), dim))
        for token, vector in zip(vocab, syn0):
            word = token.word
            vector_str = ' '.join([str(s) for s in vector])
            fo.write('%s %s\n' % (word, vector_str))

    fo.close()

#customization
def sort_frequency():
    from collections import Counter

    word_counts = Counter()
    with open('social_corpus.txt') as f:
        for line in f:
            word_counts.update(line.strip().split())
    type(word_counts)
    sorted(word_counts.items(),key=lambda x:x[1],reverse=True)
    pass

if __name__ == '__main__':
    
    fi='../data/ge/social_corpus.txt'
    fo='../data/ge/result.txt'
    cbow=0
    neg=5
    dim=50
    alpha=0.01
    win=5
    min_count=1
    binary=0

    train_process(fi, fo, cbow, neg, dim, alpha, win, min_count, binary)