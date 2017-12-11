from ke_preprocess import normalized_token, read_file
from ke_pagerank import wpr
from ke_postprocess import get_phrases
import os


def evaluate_extraction(dataset, method_name, ngrams=2, damping=0.85, omega=None, phi=None,
                        alter_topn=None, alter_edge=None, alter_node=None):
    """
    评价实验结果

    omega,phi, [0]代表不适用任何特征，权重设置为1。None为所有特征的简单加和。[-1]只用最后一个特征。
    """
    if dataset == 'KDD':
        abstr_dir = '../dataset/KDD/abstracts/'
        out_dir = '../result/'
        gold_dir = '../dataset/KDD/gold/'
        edge_dir = '../dataset/KDD/edge_features/'
        node_dir = '../dataset/KDD/node_features/'
        file_names = read_file('../dataset/KDD_filelist').split(',')
        topn = 4
    elif dataset == 'WWW':
        abstr_dir = '../dataset/WWW/abstracts/'
        out_dir = '../result/'
        gold_dir = '../dataset/WWW/gold/'
        edge_dir = '../dataset/WWW/edge_features/'
        node_dir = '../dataset/WWW/node_features/'
        file_names = read_file('../dataset/WWW_filelist').split(',')
        topn = 5
    else:
        print('wrong dataset name')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if alter_edge:
        edge_dir = alter_edge
    if alter_node:
        node_dir = alter_node
    if alter_topn:
        topn = alter_topn

    count = 0
    gold_count = 0
    extract_count = 0
    mrr = 0
    prcs_micro = 0
    recall_micro = 0
    for file_name in file_names:
        # print(file_name)
        pr, graph = wpr(edge_dir + file_name, node_dir +
                        file_name, omega=omega, phi=phi, d=damping)

        gold = read_file(gold_dir + file_name)
        keyphrases = get_phrases(pr, graph, abstr_dir, file_name, ng=ngrams)
        top_phrases = []
        for phrase in keyphrases:
            if phrase[0] not in str(top_phrases):
                top_phrases.append(phrase[0])
            if len(top_phrases) == topn:
                break
        golds = gold.split('\n')
        if golds[-1] == '':
            golds = golds[:-1]
        golds = list(' '.join(list(normalized_token(w)
                                   for w in g.split())) for g in golds)
        count_micro = 0
        position = []
        for phrase in top_phrases:
            if phrase in golds:
                count += 1
                count_micro += 1
                position.append(top_phrases.index(phrase))
        if position != []:
            mrr += 1 / (position[0] + 1)
        gold_count += len(golds)
        extract_count += len(top_phrases)
        prcs_micro += count_micro / len(top_phrases)
        recall_micro += count_micro / len(golds)

    prcs = count / extract_count
    recall = count / gold_count
    f1 = 2 * prcs * recall / (prcs + recall)
    mrr /= len(file_names)
    prcs_micro /= len(file_names)
    recall_micro /= len(file_names)
    f1_micro = 2 * prcs_micro * recall_micro / (prcs_micro + recall_micro)
    print(prcs, recall, f1, mrr)

    tofile_result = method_name + ',' + str(prcs) + ',' + str(recall) + ',' + str(f1) + ',' + str(mrr) + ',' \
        + str(prcs_micro) + ',' + str(recall_micro) + \
        ',' + str(f1_micro) + ',\n'
    with open(out_dir + dataset + '_RESULTS.csv', mode='a', encoding='utf8') as f:
        f.write(tofile_result)


if __name__ == '__main__':
    with open('../result/KDD/words.emb') as f:
        f.read()
