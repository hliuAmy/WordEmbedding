# Co-occurrence relations
from re import match
import networkx as nx
import csv
#import matplotlib.pyplot as plt
from get_tfidfByfile import get_word_tfidf_Byfile, filenames, fileTextList


def isword(word):
    return match(r'^[A-Za-z].+', word)


def edgeW(dataset, filename, text, c):
    # word and tfidf
    words = [word for word in text.split() if isword(word)]
    wordlist = set(words)
    word_tfidf = get_word_tfidf_Byfile(dataset, filename, wordlist)
    # co-cccerrence
    wordsIndex = [(word, index) for index, word in enumerate(words)]
    edgeWithlen = []
    for i in range(len(wordsIndex)):
        context_start = max(i - c, 0)
        context_end = min(i + c + 1, len(wordsIndex))
        contexts = wordsIndex[context_start:i] + wordsIndex[i + 1:context_end]
        for word in contexts:
            edgeWithlen.append(
                (wordsIndex[i][0], word[0], float(word_tfidf[wordsIndex[i][0]]) / abs(wordsIndex[i][1] - word[1])))
    wordsG = nx.DiGraph()
    for edge in edgeWithlen:
        if((edge[0], edge[1])) in wordsG.edges():
            w = wordsG[edge[0]][edge[1]]["weight"]
            wordsG.remove_edge(edge[0], edge[1])
            newEdge = (edge[0], edge[1], edge[2] + w)
            wordsG.add_weighted_edges_from([newEdge])
        else:
            wordsG.add_weighted_edges_from([edge])
    # print(len(wordsG.edges()))
    return wordsG


def wordsG(datasets=['KDD', 'WWW'], c=1):
    '''
    c=1,means windows=3
    '''
    for dataset in datasets:
        filenamesPath = './data_temp/' + dataset + '/abstractsNames'
        datasetText = './data_temp/' + dataset + '/abstracts.data'
        resultPath = './result_graph/' + dataset + '/wordsG.data'

        filenamelist = filenames(filenamesPath)
        datalist = fileTextList(datasetText)
        wordsGAll = nx.DiGraph()

        # get words'graph
        for i in range(len(filenamelist)):
            print(str(i) + "st file")
            filename = filenamelist[i]
            text = datalist[i]
            wordsG = edgeW(dataset, filename, text, c)
            for edge in wordsG.edges():
                if edge in wordsGAll.edges():
                    w = wordsGAll[edge[0]][edge[1]]["weight"] + \
                        wordsG[edge[0]][edge[1]]["weight"]
                    wordsGAll.remove_edge(edge[0], edge[1])
                    newEdge = (edge[0], edge[1], w)
                    wordsGAll.add_weighted_edges_from([newEdge])
                else:
                    wordsGAll.add_weighted_edges_from(
                        [(edge[0], edge[1], wordsG[edge[0]][edge[1]]["weight"])])
        print("add wout")
        for edge in wordsGAll.edges():
            w_out = wordsGAll.out_degree(edge[0], weight="weight")
            w = wordsGAll[edge[0]][edge[1]]["weight"]
            finalW = w / w_out
            wordsGAll.remove_edge(edge[0], edge[1])
            wordsGAll.add_weighted_edges_from([(edge[0], edge[1], finalW)])

        # nx.draw(wordsGAll)
        # plt.savefig("wordsG.png")
        # plt.show
        print(len(wordsGAll.edges()))
        print("write to file")
        with open(resultPath, mode='w', encoding='utf-8') as f:
            csvWriter = csv.writer(f)
            for edge in wordsGAll.edges():
                csvWriter.writerow(
                    [edge[0], edge[1], wordsGAll[edge[0]][edge[1]]["weight"]])
            f.close()


if __name__ == '__main__':
    wordsG(c=1)
