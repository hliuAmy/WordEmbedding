import os
import networkx as nx
import random
import math
from re import match
import csv
from get_tfidfByfile import filenames


def probability(topicNum, wordID, p):
    pro = p[topicNum].split()[int(wordID)]
    return float(pro)


def topicG(datasets=['KDD', 'WWW']):
    for dataset in datasets:
        twp_MatrixPath = './data_temp/' + dataset + '/LDA/model-final.phi'
        document_topic_Map = './data_temp/' + dataset + '/LDA/model-final.tassign'
        word_indexPath = './data_temp/' + dataset + '/LDA/wordmap.txt'
        filelistPath = './data_temp/' + dataset + '/abstractsNames'
        resultGPath = './result_graph/' + dataset + '/topicG.data'

        # data preparation
        # get t-w Matrix,the value is p(w/t)
        with open(twp_MatrixPath, mode='r', encoding="utf-8") as f:
            twp = f.readlines()
        # get w:t by file
        with open(document_topic_Map, mode='r', encoding="utf-8") as f:
            dwt = f.readlines()
        # get filelist
        filelist = filenames(filelistPath)
        #{id:word}
        id2word = {}
        word2id = {}
        with open(word_indexPath, mode='r', encoding="utf-8") as f:
            word_ids = f.readlines()
        for word_id in word_ids:
            items = word_id.split()
            if len(items) < 2:
                continue
            id2word[int(items[1])] = items[0]
            word2id[items[0]] = int(items[1])

        # topicG
        topicGAll = nx.DiGraph()
        for i in range(len(filelist)):
            print(str(i) + "st file")
            filename = filelist[i]
            fileText = dwt[i].strip('\n').strip()
            unweightedEdges = [edge.split(
                ':') for edge in fileText.split() if len(edge) != 0]
            weightedEdges = []
            for edge in unweightedEdges:
                wordID = int(edge[0])
                topic = int(edge[1])
                weight = probability(topic, wordID, twp)
                word = id2word[wordID]
                topic = "T" + str(topic)
                weightedEdges.append((word, topic, weight))
            topicGAll.add_weighted_edges_from(weightedEdges)

        # calculate lemda word_i by topicG
        print("calculate lemda word_i by topicG")
        edgesNum = len(topicGAll.nodes())
        print("number of nodes is " + str(edgesNum))
        for node in topicGAll.nodes():
            if match(r'^T[0-9]+', node):
                continue
            else:
                lemda = 0.0
                topics = topicGAll.neighbors(node)
                for topic in topics:
                    weight = topicGAll[node][topic]['weight']
                    while(True):
                        random_word = random.choice(topicGAll.nodes())
                        if match(r'^T[0-9]+', random_word):
                            continue
                        else:
                            break
                    wordID = word2id[random_word]
                    topicID = int(topic.strip('T'))
                    random_weight = probability(topicID, wordID, twp)
                    lemda += weight * math.log(weight / random_weight)
                topicGAll.add_node(node, lamda=lemda)

        # modify every edge=lemda*p
        print("modify every edge=lemda*p")
        edgesNum = len(topicGAll.edges())
        print("number of edges is " + str(edgesNum))
        for edge in topicGAll.edges():
            nodeW = topicGAll.node[edge[0]]['lamda']
            edgeW = topicGAll[edge[0]][edge[1]]['weight']
            topicGAll[edge[0]][edge[1]]['weight'] = nodeW * edgeW

        # sum of weights
        # sumofweights = 0.0
        # for u, v, d in topicGAll.edges(data=True):
        #     sumofweights += d['weight']
        # print("sum of edges: " + str(sumofweights))
        # for edge in topicGAll.edges():
        #     oldWeight = topicGAll[edge[0]][edge[1]]['weight']
        #     topicGAll[edge[0]][edge[1]]['weight'] = oldWeight / sumofweights
        weights = []
        for u, v, d in topicGAll.edges(data=True):
            weights.append(d['weight'])
        maxweight = max(weights)
        minweight = min(weights)
        # update every edges'weight
        for edge in topicGAll.edges():
            oldWeight = topicGAll[edge[0]][edge[1]]['weight']
            newWeight = (oldWeight - minweight) / (maxweight - minweight)
            topicGAll[edge[0]][edge[1]]['weight'] = newWeight

        # save topicG
        print("save topicG")
        print("write to file")
        with open(resultGPath, mode='w', encoding='utf-8') as f:
            csvWriter = csv.writer(f)
            for edge in topicGAll.edges():
                csvWriter.writerow(
                    [edge[0], edge[1], topicGAll[edge[0]][edge[1]]["weight"]])
            f.close()


if __name__ == '__main__':
    topicG()
