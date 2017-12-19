# embTrain.py
import random
import math
import numpy as np
import networkx as nx
import csv


class updateT:

    def __init__(self, wtG, wordsVec, dim):
        self.wtG = wtG
        self.dim = dim
        if wordsVec == None:
            self.wordsVec = {}
        else:
            self.wordsVec = wordsVec
        self.topicsVec = {}
        self.edges_weight_t = []
        self.edges_t = []
        self.nodes_weight_t = {}
        self.sigmoid_table = []
        self.alias_t = []
        self.prob_t = []
        self.neg_table_t = []
        self.SIGMOID_BOUND = 6
        self.sigmoid_table_size = 1000
        self.neg_table_size = 1000000
        self.init_rho = 0.025
        self.rho = 0.0
        self.num_negative = 5
        self.M = np.zeros((self.dim, self.dim))
        #self.M = np.eye(self.dim)

    def readG(self):
        for u, v, d in self.wtG.edges(data=True):
            w = float(d['weight']) * 10
            self.edges_weight_t.append(w)
            self.edges_t.append((u, v, w))
            if v in self.nodes_weight_t.keys():
                self.nodes_weight_t[v] += w
            else:
                self.nodes_weight_t[v] = w
        if len(self.wordsVec.keys()) == 0:
            for node in self.wtG.nodes():
                self.wordsVec[node] = np.random.uniform(
                    low=-0.5 / self.dim, high=0.5 / self.dim, size=(self.dim))
        for node in self.wtG.nodes():
            if node.startswith('T'):
                self.topicsVec[node] = np.random.uniform(
                    low=-0.5 / self.dim, high=0.5 / self.dim, size=(self.dim))

    def initAliasTable_t(self):
        length = len(self.edges_weight_t)
        norm_prob = []
        large_block = []
        small_block = []
        self.prob_t = [0.0] * length
        self.alias_t = [0.0] * length
        num_small_block = 0
        num_large_block = 0
        sum_weight = sum(self.edges_weight_t)
        for i in range(length):
            norm_prob.append(self.edges_weight_t[i] * length / sum_weight)
        for i in range(length)[::-1]:
            if norm_prob[i] < 1:
                small_block.append(i)
                num_small_block += 1
            else:
                large_block.append(i)
                num_large_block += 1
        while num_large_block > 0 and num_small_block > 0:
            num_small_block -= 1
            num_large_block -= 1
            cur_small_block = small_block[num_small_block]
            cur_large_block = large_block[num_large_block]
            self.prob_t[cur_small_block] = norm_prob[cur_small_block]
            self.alias_t[cur_small_block] = cur_large_block

            large_prob = norm_prob[cur_large_block]
            small_prob = norm_prob[cur_small_block]
            norm_prob[cur_large_block] = large_prob + small_prob - 1
            if(norm_prob[cur_large_block] < 1):
                small_block[num_small_block] = cur_large_block
                num_small_block += 1
            else:
                large_block[num_large_block] = cur_large_block
                num_large_block += 1
        while(num_large_block > 0):
            num_large_block -= 1
            index = large_block[num_large_block]
            self.prob_t[index] = 1.0
        while(num_small_block > 0):
            num_small_block -= 1
            index = small_block[num_small_block]
            self.prob_t[index] = 1.0
        del norm_prob
        del small_block
        del large_block

    def sampleAnEdge_t(self, rand_value1, rand_value2):
        k = len(self.edges_weight_t) * rand_value1
        if rand_value2 < self.prob_t[int(k)]:
            return k
        else:
            return self.alias_t[int(k)]

    def initNegTable_t(self):
        ssum = 0.0
        cur_sum = 0.0
        por = 0.0
        i = 0
        neg_sampling_power = 0.75
        for value in self.nodes_weight_t.values():
            try:
                ssum += math.pow(value, neg_sampling_power)
            except ValueError as err:
                print(value)
        for word in self.nodes_weight_t.keys():
            cur_sum += math.pow(self.nodes_weight_t[word], neg_sampling_power)
            por = cur_sum / ssum
            while i < self.neg_table_size and float(i) / self.neg_table_size < por:
                self.neg_table_t.append(word)
                i += 1

    def initSigmoidTable(self):
        self.sigmoid_table = np.zeros(self.neg_table_size)
        for i in range(self.sigmoid_table_size):
            x = 2 * self.SIGMOID_BOUND * i / self.sigmoid_table_size - self.SIGMOID_BOUND
            self.sigmoid_table[i] = 1 / (1 + math.exp(-x))

    def FastSigmoid(self, x):
        if x > self.SIGMOID_BOUND:
            return 1
        elif x < -self.SIGMOID_BOUND:
            return 0
        else:
            k = int((x + self.SIGMOID_BOUND) *
                    self.sigmoid_table_size / self.SIGMOID_BOUND / 2)
            return self.sigmoid_table[k]

    def initial(self):
        self.readG()
        self.initAliasTable_t()
        self.initNegTable_t()
        self.initSigmoidTable()
        print('init is ok')

    def train(self, total_iter, outputPath):
        for i in range(total_iter):
            if i % 100000 == 0:
                print(i)
            if i % 1000000 == 0:
                self.rho = self.init_rho * (1.0 - float(i) / total_iter)
                if self.rho < self.init_rho * 0.0001:
                    self.rho = self.init_rho * 0.0001
                print(i, ':', self.rho)
            # self.trainT_noM()
            self.trainT()
            if i == total_iter - 1:
                self.output(outputPath, self.wordsVec)

    def trainT(self):
        vec_error = np.zeros(self.dim)
        M_error = np.zeros((self.dim, self.dim))

        def choiceEdge():
            random1 = np.random.random()
            random2 = np.random.random()
            edgeID = int(self.sampleAnEdge_t(random1, random2))
            edge = self.edges_t[edgeID]
            return edge
        edge = choiceEdge()
        while edge[0] not in self.wordsVec.keys():
            edge = choiceEdge()
        u = edge[0]
        v = edge[1]
        w = float(edge[2])
        label = 0
        target = ''
        for i in range(self.num_negative + 1):
            if i == 0:
                label = 1
                target = v
            else:
                neg_index = int(self.neg_table_size * np.random.random())
                target = self.neg_table_t[neg_index]
                if u == None or v == None or target == None:
                    print(u, v, neg_index, target)
                if target == u or target == v:
                    i -= 1
                    continue
                label = 0
            x = np.dot(
                np.dot(self.wordsVec[u].T, self.M), self.topicsVec[target])
            g = (label - self.FastSigmoid(x)) * self.rho
            vec_error = vec_error + g * np.dot(self.M, self.topicsVec[target])
            M_error = M_error + g * \
                np.dot(self.wordsVec[u], self.topicsVec[target].T)
            self.topicsVec[target] += g * np.dot(self.M, self.wordsVec[u])
        self.wordsVec[u] = self.wordsVec[u] + vec_error
        self.M = self.M + M_error

    def trainT_noM(self):
        vec_error = np.zeros(self.dim)

        def choiceEdge():
            random1 = np.random.random()
            random2 = np.random.random()
            edgeID = int(self.sampleAnEdge_t(random1, random2))
            edge = self.edges_t[edgeID]
            return edge
        edge = choiceEdge()
        while edge[0] not in self.wordsVec.keys():
            edge = choiceEdge()
        u = edge[0]
        v = edge[1]
        w = float(edge[2])
        label = 0
        target = ''
        for i in range(self.num_negative + 1):
            if i == 0:
                label = 1
                target = v
            else:
                neg_index = int(self.neg_table_size * np.random.random())
                target = self.neg_table_t[neg_index]
                if u == None or v == None or target == None:
                    print(u, v, neg_index, target)
                if target == u or target == v:
                    i -= 1
                    continue
                label = 0
            x = np.dot(self.wordsVec[u], self.topicsVec[target])
            g = (label - self.FastSigmoid(x)) * self.rho
            vec_error = vec_error + g * self.topicsVec[target]
            self.topicsVec[target] += g * self.wordsVec[u]
        self.wordsVec[u] = self.wordsVec[u] + vec_error

    def output(self, path, wordsVec):
        with open(path, mode='w', encoding='utf-8')as f:
            csvWriter = csv.writer(f)
            for key, value in wordsVec.items():
                row = []
                row.append(key)
                for item in value:
                    row.append(float(item))
                row[1:] = self.regularlizationVec(row[1:])
                csvWriter.writerow(row)
            f.close()

    def regularlizationVec(self, vec):
        from sklearn import preprocessing
        vec1 = []
        vec1.append(vec)
        vec1 = preprocessing.normalize(vec1, 'l2')
        return vec1[0]

    def normalizationVec(self, vec):
        minvalue = min(vec)
        maxvalue = max(vec)
        normvalue = maxvalue - minvalue
        for i in range(len(vec)):
            vec[i] = (vec[i] - minvalue) / normvalue
        return vec

    def avgVec(self, G, wordsVec):
        for word in G.nodes():
            context = G.neighbors(word)
            avg = wordsVec[word]
            for c in context:
                avg = avg + wordsVec[c]
            avg = avg / len(context)
            wordsVec[word] = avg
        path = '../result/KDD/words.emb_avg_tf_count_noM'
        self.output(path, wordsVec)
