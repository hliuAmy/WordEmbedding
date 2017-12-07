# Co-occurrence relations
from re import match
import networkx as nx
import csv
# import matplotlib.pyplot as plt
from get_tfidfByfile import get_word_tfidf_Byfile, filenames, fileTextList


def isword(word):
    return match(r'^[A-Za-z].+', word)


def wordTF(datasetlist):
    word_tf = {}
    wordsAll = []
    for text in datasetlist:
        words = [word for word in text.split() if isword(word)]
        wordsAll.extend(words)
    for word in wordsAll:
        if word not in word_tf.keys():
            word_tf[word] = wordsAll.count(word)
    return word_tf


def edgeW_tf(dataset, filename, text, c, word_tf):
    # word
    words = [word for word in text.split() if isword(word)]
    # co-cccerrence
    wordsIndex = [(word, index) for index, word in enumerate(words)]
    edgeWithlen = []
    for i in range(len(wordsIndex)):
        context_start = max(i - c, 0)
        context_end = min(i + c + 1, len(wordsIndex))
        contexts = wordsIndex[context_start:i] + wordsIndex[i + 1:context_end]
        for word in contexts:
            edgeWithlen.append(
                (wordsIndex[i][0], word[0], float(word_tf[wordsIndex[i][0]]) / abs(wordsIndex[i][1] - word[1])))
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


def edgeW_tdidf(dataset, filename, text, c):
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


def edgeW(dataset, filename, text, c):
    # word and tfidf
    words = [word for word in text.split() if isword(word)]
    # co-cccerrence
    wordsIndex = [(word, index) for index, word in enumerate(words)]
    edgeWithlen = []
    for i in range(len(wordsIndex)):
        context_start = max(i - c, 0)
        context_end = min(i + c + 1, len(wordsIndex))
        contexts = wordsIndex[context_start:i] + wordsIndex[i + 1:context_end]
        for word in contexts:
            edgeWithlen.append(
                (wordsIndex[i][0], word[0], float(1) / abs(wordsIndex[i][1] - word[1])))
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
        resultPath = './result_graph/' + dataset + '/wordsG_tf.data'

        filenamelist = filenames(filenamesPath)
        datalist = fileTextList(datasetText)
        word_tf = wordTF(datalist)
        wordsGAll = nx.DiGraph()

        # get words'graph
        for i in range(len(filenamelist)):
            print(str(i) + "st file")
            filename = filenamelist[i]
            text = datalist[i]
            wordsG = edgeW_tf(dataset, filename, text, c, word_tf)
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


def edgeW_count(dataset, filename, text, c):
    # word and tfidf
    words = [word for word in text.split() if isword(word)]
    # co-cccerrence
    wordsIndex = [(word, index) for index, word in enumerate(words)]
    edgeWithlen = []
    edgeWithCount = []
    for i in range(len(wordsIndex)):
        context_start = max(i - c, 0)
        context_end = min(i + c + 1, len(wordsIndex))
        contexts = wordsIndex[context_start:i] + wordsIndex[i + 1:context_end]
        for word in contexts:
            edgeWithlen.append(
                (wordsIndex[i][0], word[0], float(1) / abs(wordsIndex[i][1] - word[1])))
    for i in range(len(wordsIndex)):
        context_start = max(i - c, 0)
        context_end = min(i + c + 1, len(wordsIndex))
        contexts = wordsIndex[context_start:i] + wordsIndex[i + 1:context_end]
        for word in contexts:
            edgeWithCount.append((wordsIndex[i][0], word[0], 1))
    wordsG = nx.DiGraph()
    wordsCount = nx.DiGraph()
    for edge in edgeWithlen:
        if((edge[0], edge[1])) in wordsG.edges():
            w = wordsG[edge[0]][edge[1]]["weight"]
            wordsG.remove_edge(edge[0], edge[1])
            newEdge = (edge[0], edge[1], edge[2] + w)
            wordsG.add_weighted_edges_from([newEdge])
        else:
            wordsG.add_weighted_edges_from([edge])
    for edge in edgeWithCount:
        if((edge[0], edge[1])) in wordsCount.edges():
            c = wordsCount[edge[0]][edge[1]]["weight"]
            wordsCount.remove_edge(edge[0], edge[1])
            newEdge = (edge[0], edge[1], edge[2] + c)
            wordsCount.add_weighted_edges_from([newEdge])
        else:
            wordsCount.add_weighted_edges_from([edge])
    # print(len(wordsG.edges()))
    return wordsG, wordsCount


def wordsG_count(datasets=['KDD', 'WWW'], c=1):
    '''
    c=1,means windows=3
    '''
    for dataset in datasets:
        filenamesPath = './data_temp/' + dataset + '/abstractsNames'
        datasetText = './data_temp/' + dataset + '/abstracts.data'
        resultPath = './result_graph/' + dataset + '/wordsG_tf_count.data'

        filenamelist = filenames(filenamesPath)
        datalist = fileTextList(datasetText)
        word_tf = wordTF(datalist)
        wordsGAll = nx.DiGraph()
        wordsCountGAll = nx.DiGraph()

        # get words'graph
        for i in range(len(filenamelist)):
            print(str(i) + "st file")
            filename = filenamelist[i]
            text = datalist[i]
            wordsG, wordsCount = edgeW_count(dataset, filename, text, c)
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
            for edge in wordsCount.edges():
                if edge in wordsCountGAll.edges():
                    w = wordsCountGAll[edge[0]][edge[1]]["weight"] + \
                        wordsCount[edge[0]][edge[1]]["weight"]
                    wordsCountGAll.remove_edge(edge[0], edge[1])
                    newEdge = (edge[0], edge[1], w)
                    wordsCountGAll.add_weighted_edges_from([newEdge])
                else:
                    wordsCountGAll.add_weighted_edges_from(
                        [(edge[0], edge[1], wordsCount[edge[0]][edge[1]]["weight"])])
        print("add wout")
        for edge in wordsGAll.edges():
            w_out = wordsGAll.out_degree(edge[0], weight="weight")
            count = wordsCountGAll[edge[0]][edge[1]]["weight"]
            w_tf = word_tf[edge[0]]
            w = wordsGAll[edge[0]][edge[1]]["weight"]
            finalW = w * w_tf / (w_out * count)
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
    wordsG_count(c=1)
