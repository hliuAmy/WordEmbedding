# embTrain.py
import random
import math
import numpy as np
import networkx as nx
import csv


class embTrain:

    def __init__(self, G, dim=128, labelG='w'):
        self.G = G
        self.dim = dim
        self.labelG = labelG
        self.wordsVec = {}
        self.contextVec = {}
        self.edges_weight = []
        self.edges = []
        self.nodes_weight = {}
        self.sigmoid_table = []
        self.alias = []
        self.prob = []
        self.neg_table = []
        self.SIGMOID_BOUND = 6
        self.sigmoid_table_size = 1000
        self.neg_table_size = 1000000
        self.init_rho = 0.05
        self.rho = 0.0
        self.num_negative = 5

    def readG(self):
        if self.labelG == 'w':
            for u, v, d in self.G.edges(data=True):
                w = float(d['weight'])
                self.edges_weight.append(w)
                self.edges.append((u, v, w))
                if u in self.nodes_weight.keys():
                    self.nodes_weight[u] += w
                else:
                    self.nodes_weight[u] = w
            for node in self.G.nodes():
                self.wordsVec[node] = np.random.uniform(
                    low=-0.5 / self.dim, high=0.5 / self.dim, size=(self.dim))
                self.contextVec[node] = np.zeros(self.dim)
        else:
            for u, v, d in self.G.edges(data=True):
                w = float(d['weight']) * 10
                self.edges_weight.append(w)
                self.edges.append((u, v, w))
                if v in self.nodes_weight.keys():
                    self.nodes_weight[v] += w
                else:
                    self.nodes_weight[v] = w
            for node in self.G.nodes():
                if node.startswith('T'):
                    self.contextVec[node] = np.zeros(self.dim)

    def initAliasTable(self):
        length = len(self.edges_weight)
        norm_prob = []
        large_block = []
        small_block = []
        self.prob = [0.0] * length
        self.alias = [0.0] * length
        num_small_block = 0
        num_large_block = 0
        sum_weight = sum(self.edges_weight)
        for i in range(length):
            norm_prob.append(self.edges_weight[i] * length / sum_weight)
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
            self.prob[cur_small_block] = norm_prob[cur_small_block]
            self.alias[cur_small_block] = cur_large_block

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
            self.prob[index] = 1.0
        while(num_small_block > 0):
            num_small_block -= 1
            index = small_block[num_small_block]
            self.prob[index] = 1.0
        del norm_prob
        del small_block
        del large_block

    def sampleAnEdge(self, rand_value1, rand_value2):
        k = len(self.edges_weight) * rand_value1
        if rand_value2 < self.prob[int(k)]:
            return k
        else:
            return self.alias[int(k)]

    def initNegTable(self):
        ssum = 0.0
        cur_sum = 0.0
        por = 0.0
        i = 0
        neg_sampling_power = 0.75
        for value in self.nodes_weight.values():
            try:
                ssum += math.pow(value, neg_sampling_power)
            except ValueError as err:
                print(value)
        for word in self.nodes_weight.keys():
            cur_sum += math.pow(self.nodes_weight[word], neg_sampling_power)
            por = cur_sum / ssum
            while i < self.neg_table_size and float(i) / self.neg_table_size < por:
                self.neg_table.append(word)
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
        self.initAliasTable()
        self.initNegTable()
        self.initSigmoidTable()
        print('ok')

    def trainW(self, total_iter):
        for i in range(total_iter):

            if i % 1000000 == 0:
                self.rho = self.init_rho * (1.0 - float(i) / total_iter)
                if self.rho < self.init_rho * 0.0001:
                    self.rho = self.init_rho * 0.0001
                print(i, ':', self.rho)
            vec_error = np.zeros(self.dim)
            random1 = np.random.random()
            random2 = np.random.random()
            edgeID = int(self.sampleAnEdge(random1, random2))
            edge = self.edges[edgeID]
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
                    target = self.neg_table[neg_index]
                    if u == None or v == None or target == None:
                        print(u, v, neg_index, target)
                    if target == u or target == v:
                        i -= 1
                        continue
                    label = 0
                x = np.dot(self.wordsVec[u], self.contextVec[target])
                g = (label - self.FastSigmoid(x)) * self.rho
                vec_error += g * self.contextVec[target]
                self.contextVec[target] += g * self.wordsVec[u]
            self.wordsVec[u] += vec_error

    def trainT(self, total_iter, wordsVec, outputPath):
        M = np.zeros((self.dim, self.dim))
        for j in range(total_iter):
            if j % 1000000 == 0:
                self.rho = self.init_rho * (1.0 - float(j) / total_iter)
                if self.rho < self.init_rho * 0.0001:
                    self.rho = self.init_rho * 0.0001
                print(j, ':', self.rho)
            vec_error = np.zeros(self.dim)
            M_error = np.zeros((self.dim, self.dim))
            random1 = np.random.random()
            random2 = np.random.random()
            edgeID = int(self.sampleAnEdge(random1, random2))
            edge = self.edges[edgeID]
            u = edge[0]
            v = edge[1]
            w = float(edge[2])
            label = 0
            target = ''
            if u not in wordsVec.keys():
                j -= 1
                continue
            for i in range(self.num_negative + 1):
                if i == 0:
                    label = 1
                    target = v
                else:
                    neg_index = int(self.neg_table_size * np.random.random())
                    target = self.neg_table[neg_index]
                    if u == None or v == None or target == None:
                        print(u, v, neg_index, target)
                    if target == u or target == v:
                        i -= 1
                        continue
                    label = 0
                x = np.dot(np.dot(wordsVec[u], M), self.contextVec[target])
                g = (label - self.FastSigmoid(x)) * self.rho
                vec_error += g * np.dot(self.contextVec[target], M)
                M_error += g * np.dot(wordsVec[u].T, self.contextVec[target])
                self.contextVec[target] += g * np.dot(wordsVec[u].T, M.T)
            wordsVec[u] += vec_error
            M += M_error
        self.output(outputPath, wordsVec)

    def trainINE(self):
        pass

    def output(self, path, wordsVec):
        with open(path, mode='w', encoding='utf-8')as f:
            csvWriter = csv.writer(f)
            for key, value in wordsVec.items():
                row = []
                row.append(key)
                for item in value:
                    row.append(float(item))
                csvWriter.writerow(row)
            f.close()
