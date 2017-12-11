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


def main(total_iter, dim, topicNum, dataset,alpha):
    '''
    total_iter:number of sample edge
    '''
    input_wordsG_path = '../data_preparation/result_graph/' + \
        dataset + '/wordsG_tf_count.data'
    input_topicG_path = '../data_preparation/result_graph/' + dataset + '/topicG.data'
    words_path = '../result/' + dataset + '/onlywords.emb_noavg_tf_count_noM3'
    emb_words_path = '../result/' + dataset + '/words.emb_noavg_tf_count_noM3'

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

    print("*****train*****")
    trainG.trainW(total_iter,alpha)
    trainG.output(words_path, trainG.wordsVec)
    trainGT.trainT_noM(total_iter, trainG.wordsVec, emb_words_path, trainG.G,alpha)


if __name__ == '__main__':
    datasets = ['KDD']
    for dataset in datasets:
        main(total_iter=10000000, dim=200, topicNum=50, dataset=dataset,alpha=0.5)
