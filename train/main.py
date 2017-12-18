# main
import networkx as nx
from embTrain import *


def readFile(path):
    G = nx.DiGraph()
    with open(path, mode='r', encoding='utf-8') as f:
        text = f.read()
    edges = []
    for line in text.split('\n'):
        if len(line.strip()) == 0:
            continue
        elif len(line.split(',')) != 3:
            continue
        else:
            edge = tuple(line.split(','))
            if 'a' in edge:
                continue
            else:
                edges.append(edge)
    G.add_weighted_edges_from(edges)
    return G


def combinationVec(sourceVec, distinationVec, alpha):
    result = {}
    for key in sourceVec:
        result[key] = alpha * sourceVec[key] + \
            (1 - alpha) * distinationVec[key]
    return result


def main(total_iter, dim, topicNum, dataset, alpha):
    # '''
    # total_iter:number of sample edge
    # '''
    input_wordsG_path = '../data_preparation/result_graph/' + \
        dataset + '/wordsG_tf_count.data'
    input_topicG_path = '../data_preparation/result_graph/' + dataset + '/topicG.data'
    emb_words_path = '../result/' + dataset + '/onlywords.emb_noM'
    emb_topic_words_path = '../result/' + dataset + '/words.emb_noM'
    emb_cmb_words_path = '../result/' + dataset + '/cmb_words.emb_noM_'

    print("*****Read Data*****")
    wordsG = readFile(input_wordsG_path)
    print(dataset + "'s wordsG's number of edges is ", len(wordsG.edges()))
    topicG = readFile(input_topicG_path)
    print(dataset + "'s topicG's number of edges is ", len(topicG.edges()))

    print("*****init for train*****")
    trainG = embTrain(wordsG, dim, 'w')
    trainG.initial()
    trainGT = embTrain(topicG, dim, 't')
    trainGT.initial()
    trainG_temp = embTrain(wordsG, dim, 'w')
    trainG_temp.initial()

    print("*****train*****")
    trainG.trainW(total_iter)
    trainG.output(emb_words_path, trainG.wordsVec)
    wordsVecT = trainGT.trainT_noM(total_iter, trainG_temp.wordsVec, trainG.G)
    trainGT.output(emb_topic_words_path, wordsVecT)

    print('******combination*****')
    for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        cmbVec = combinationVec(trainG.wordsVec, wordsVecT, alpha)
        trainG.output(emb_cmb_words_path + str(alpha), cmbVec)

    # try tf or tfidf for wordsG;we test five times for embedding
    # input_wordsG_path = '../data_preparation/result_graph/' +  dataset + '/wordsG_tfidf.data'
    # emb_words_path = '../result/' + dataset + '/emb_tfidf'
    # wordsG = readFile(input_wordsG_path)
    # print(dataset + "'s wordsG's number of edges is ", len(wordsG.edges()))
    # for i in range(5):
    #     trainG = embTrain(wordsG, dim, 'w')
    #     trainG.initial()
    #     trainG.trainW(total_iter)
    #     trainG.output(emb_words_path+str(i), trainG.wordsVec)


if __name__ == '__main__':
    datasets = ['KDD']
    for dataset in datasets:
        main(total_iter=1000000, dim=200, topicNum=50, dataset=dataset, alpha=0.5)
