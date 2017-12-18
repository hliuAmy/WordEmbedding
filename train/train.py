
# main
import networkx as nx
from updateW import *
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


def main(total_iter, dim, dataset,alpha,beta):
    # '''
    # total_iter:number of sample edge
    # '''
    input_wordsG_path = '../data_preparation/result_graph/' + \
        dataset + '/wordsG_tf_count.data'
    input_topicG_path = '../data_preparation/result_graph/' + dataset + '/topicG.data'
    result_path = '../result/' + dataset + '/emb_beta_'+str(beta)

    print("*****Read Data*****")
    wG = readFile(input_wordsG_path)
    print(dataset + "'s wordsG's number of edges is ", len(wG.edges()))
    wtG = readFile(input_topicG_path)
    print(dataset + "'s topicG's number of edges is ", len(wtG.edges()))

    print("*****train*****")
    trainG = updateW(wG, dim)
    trainG.initial()
    trainG.trainW(total_iter)
    trainGT = embTrain(wG, wtG, trainG.wordsVec, dim)
    trainGT.initial()
    trainGT.train(total_iter, result_path,alpha,beta)


if __name__ == '__main__':
    datasets = ['KDD']
    for dataset in datasets:
        for i in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
            main(total_iter=1000000, dim=200, dataset=dataset,alpha=0,beta=i)
