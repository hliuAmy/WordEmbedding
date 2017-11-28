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


def main(total_iter, dim, topicNum, dataset):
    '''
    total_iter:number of sample edge
    '''
    input_wordsG_path = '../data_preparation/result_graph/' + dataset + '/wordsG.data'
    input_topicG_path = '../data_preparation/result_graph/' + \
        dataset + '/topicG.data'
    emb_words_path = '../result/' + dataset + '/words.emb'

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
    trainG.trainW(total_iter)
    trainGT.trainT(total_iter, trainG.wordsVec, emb_words_path)


if __name__ == '__main__':
    datasets = ['KDD', 'WWW']
    for dataset in datasets:
        main(total_iter=100000000, dim=128, topicNum=50, dataset=dataset)
