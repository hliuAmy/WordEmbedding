# embTrain.py
import random
import math
import numpy as np
import networkx as nx
import csv


class embTrain:

    def __init__(self, wG, wtG, wordsVec, topicsVec, M, dim):
        self.wG = wG
        self.wtG = wtG
        self.dim = dim
        if wordsVec == None:
            self.wordsVec = {}
        else:
            self.wordsVec = wordsVec
        if topicsVec == None:
            self.topicsVec = {}
        else:
            self.topicsVec = topicsVec
        self.contextVec = {}
        self.edges_weight = []
        self.edges = []
        self.nodes_weight = {}
        self.edges_weight_t = []
        self.edges_t = []
        self.nodes_weight_t = {}
        self.sigmoid_table = []
        self.alias = []
        self.prob = []
        self.alias_t = []
        self.prob_t = []
        self.neg_table_t = []
        self.neg_table = []
        self.SIGMOID_BOUND = 6
        self.sigmoid_table_size = 1000
        self.neg_table_size = 1000000
        self.init_rho = 0.001
        self.rho = 0.001
        self.num_negative = 5
        self.M = M
        self.loss = 0.0
        self.converge = False

    def readG(self):
        for u, v, d in self.wG.edges(data=True):
            w = float(d['weight'])
            self.edges_weight.append(w)
            self.edges.append((u, v, w))
            if u in self.nodes_weight.keys():
                self.nodes_weight[u] += w
            else:
                self.nodes_weight[u] = w
        for node in self.wG.nodes():
            self.contextVec[node] = np.zeros(self.dim)
        if len(self.wordsVec.keys()) == 0:
            for node in self.wG.nodes():
                if node.startswith('T'):
                    self.wordsVec[node] = np.random.uniform(
                        low=-0.5 / self.dim, high=0.5 / self.dim, size=(self.dim))

        for u, v, d in self.wtG.edges(data=True):
            w = float(d['weight']) * 10
            self.edges_weight_t.append(w)
            self.edges_t.append((u, v, w))
            if v in self.nodes_weight_t.keys():
                self.nodes_weight_t[v] += w
            else:
                self.nodes_weight_t[v] = w
        if len(self.topicsVec.keys()) == 0:
            for node in self.wtG.nodes():
                if node.startswith('T'):
                    # self.topicsVec[node] = np.random.uniform(
                    #     low=-0.5 / self.dim, high=0.5 / self.dim, size=(self.dim))
                    self.topicsVec[node] = np.zeros(self.dim)

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
        self.initAliasTable()
        self.initNegTable()
        self.initAliasTable_t()
        self.initNegTable_t()
        self.initSigmoidTable()
        self.loss = self.getloss()
        print('init is ok')

    def train(self, total_iter, outputPath, alpha, beta):
        for i in range(total_iter):
            if i % 100000 == 0:
                print(i)
            if i % 1000000 == 0:
                self.rho = self.init_rho * (1.0 - float(i) / total_iter)
                if self.rho < self.init_rho * 0.01:
                    self.rho = self.init_rho * 0.01
                print(i, ':', self.rho)
            self.trainW(alpha)
            self.trainT(alpha, beta)
            # self.trainT_noM()
            if i == total_iter - 1:
                self.output(outputPath, self.wordsVec)

    def train(self, total_iter, outputPath):
        print('initloss=' + str(self.loss))
        initloss = self.loss
        for i in range(total_iter):
            if self.rho < self.init_rho * 0.001:
                self.rho = self.init_rho * 0.001
            if i % 10000 == 0 and i != 0:
                curloss = self.getloss()
                # print('curloss=' + str(curloss))
                # print('rho=' + str(self.rho))
                if math.fabs(curloss - self.rho) / self.loss < 0.001:
                    self.converge = False
                elif self.loss >= curloss:
                    self.rho = self.rho * 0.5
                else:
                    self.rho = -self.rho * 0.5
                self.loss = curloss
            self.trainW()
            self.trainT()
            if i == total_iter - 1 or self.converge == True:
                print(self.loss)
                self.output(outputPath, self.wordsVec)
                break

    def getloss(self):
        loss = 0.0
        for i in range(len(self.edges_t)):
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
            a = np.dot(np.dot(self.wordsVec[u].T, self.M), self.topicsVec[v])
            b = self.FastSigmoid(a)
            if b == 0:
                b = 0.0001
            c = -math.log(b)
            loss = loss + c
            label = 0
            target = ''
            for j in range(self.num_negative):
                neg_index = int(self.neg_table_size * np.random.random())
                target = self.neg_table_t[neg_index]
                if u == None or v == None or target == None:
                    print(u, v, neg_index, target)
                if target == u or target == v:
                    j -= 1
                    continue
                value = self.FastSigmoid(
                    1 - np.dot(np.dot(self.wordsVec[u].T, self.M), self.topicsVec[target]))
                if value == 0:
                    value = 0.0001
                loss += -math.log(value)
        for i1 in range(len(self.edges)):
            def choiceEdge():
                random11 = np.random.random()
                random21 = np.random.random()
                edgeID1 = int(self.sampleAnEdge(random11, random21))
                edge1 = self.edges[edgeID1]
                return edge1
            edge1 = choiceEdge()
            while edge1[0] not in self.wordsVec.keys():
                edge1 = choiceEdge()
            u1 = edge1[0]
            v1 = edge1[1]
            value = self.FastSigmoid(
                np.dot(self.wordsVec[u1], self.contextVec[v1]))
            if value == 0:
                value = 0.0001
            loss += - math.log(value)
            target1 = ''
            for j1 in range(self.num_negative):
                neg_index1 = int(self.neg_table_size * np.random.random())
                target1 = self.neg_table[neg_index1]
                if u == None or v == None or target1 == None:
                    print(u, v, neg_index1, target1)
                if target1 == u or target1 == v:
                    j1 -= 1
                    continue
                value = self.FastSigmoid(
                    1 - np.dot(self.wordsVec[u1], self.contextVec[target1]))
                if value == 0:
                    value = 0.0001
                loss += -math.log(value)
        return loss

    def trainW(self):
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
        self.wordsVec[u] = self.wordsVec[u] + vec_error

    def trainT(self):
        vec_error = np.zeros(self.dim)
        M_error = np.zeros((self.dim, self.dim))
        # signM = np.zeros((self.dim, self.dim))
        # for m in range(self.dim):
        #     for n in range(self.dim):
        #         if self.M[m][n] > 0:
        #             signM[m][n] = 1
        #         if self.M[m][n] < 0:
        #             signM[m][n] = -1

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
        # self.M = self.M + M_error + 0.01 * signM
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
