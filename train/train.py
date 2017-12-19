
import networkx as nx
from updateW import *
from updateT import *
from updateWT import *


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


def main(total_iter, dim, dataset, alpha, beta):
    # '''
    # total_iter:number of sample edge
    # '''
    input_wordsG_path = '../data_preparation/result_graph/' + dataset + '/wordsG_tf.data'
    input_wordsG_path1 = '../data_preparation/result_graph/' + \
        dataset + '/wordsG_tf_count.data'
    input_topicG_path = '../data_preparation/result_graph/' + dataset + '/topicG100.data'
    input_topicG_path1 = '../data_preparation/result_graph/' + dataset + '/topicG.data'
    trainG_path = '../result/' + dataset + '/trainG_tf.emb'
    trainG_path1 = '../result/' + dataset + '/trainG_tf_count.emb'
    trainT_path = '../result/' + dataset + '/trainT100_noM.emb'
    trainT_path1 = '../result/' + dataset + '/trainT50_noM.emb'
    result_path = '../result/' + dataset + '/emb23'

    print("*****Read Data*****")
    wG = readFile(input_wordsG_path)
    print(dataset + "'s wordsG's number of edges is ", len(wG.edges()))
    wG1 = readFile(input_wordsG_path1)
    print(dataset + "'s wordsG's number of edges is ", len(wG1.edges()))
    wtG = readFile(input_topicG_path)
    print(dataset + "'s topicG's number of edges is ", len(wtG.edges()))
    wtG1 = readFile(input_topicG_path1)
    print(dataset + "'s topicG's number of edges is ", len(wtG1.edges()))

    print("*****train*****")
    # trainG = updateW(wG, dim)
    # trainG.initial()
    # trainG.trainW(total_iter)
    # trainG.output(trainG_path, trainG.wordsVec)
    # trainG1 = updateW(wG1, dim)
    # trainG1.initial()
    # trainG1.trainW(total_iter)
    # trainG1.output(trainG_path1, trainG1.wordsVec)

    trainT = updateT(wtG, None, dim)
    trainT.initial()
    trainT.train(total_iter, trainT_path)
    trainT1 = updateT(wtG1, None, dim)
    trainT1.initial()
    trainT1.train(total_iter, trainT_path1)

    # trainGT = embTrain(wG, wtG, trainT.wordsVec, trainT.topicsVec, dim)
    # trainGT.initial()
    # trainGT.train(total_iter, result_path, alpha, beta)


if __name__ == '__main__':
    datasets = ['KDD', 'WWW']
    for dataset in datasets:
        main(total_iter=1000000, dim=200, dataset=dataset, alpha=0, beta=0)
