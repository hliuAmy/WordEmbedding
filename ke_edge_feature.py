# ke_edge_feature
import csv
import math
import gensim

from ke_preprocess import read_file, filter_text, normalized_token
from ke_postprocess import rm_tags


def cosine_sim(vec1, vec2):
    """余弦相似度"""
    import numpy as np
    from numpy import linalg as la

    inA = np.mat(vec1)
    inB = np.mat(vec2)
    num = float(inA * inB.T)  # 若为行向量: A * B.T
    donom = la.norm(inA) * la.norm(inB)  # 余弦值
    return 0.5 + 0.5 * (num / donom)  # 归一化
    # 关于归一化：因为余弦值的范围是 [-1,+1] ，相似度计算时一般需要把值归一化到 [0,1]


def euc_distance(vec1, vec2):
    """欧式距离"""
    tmp = map(lambda x: abs(x[0] - x[1]), zip(vec1, vec2))
    distance = math.sqrt(sum(map(lambda x: x * x, tmp)))
    # distance==0时如何处理？
    if distance == 0:
        distance = 0.1
    return distance


def read_vec(path, standard=True):
    """
    read vec: word, 1, 3, 4, ....
    return word:[1,...] dict
    """
    vec_dict = {}
    with open(path, encoding='utf-8') as file:
        # 标准csv使用','隔开，有的文件使用空格，所以要改变reader中的delimiter参数
        if standard:
            table = csv.reader(file)
        else:
            table = csv.reader(file, delimiter=' ')
        for row in table:
            try:
                vec_dict[row[0]] = list(float(i) for i in row[1:])
            except:
                continue
    return vec_dict


def read_edges(path):
    """
    read csv edge features
    return a (node1, node2):[features] dict
    """
    edges = {}
    with open(path, encoding='utf-8') as file:
        table = csv.reader(file)
        for row in table:
            edges[(row[0], row[1])] = [float(i) for i in row[2:]]
    return edges


def add_word_attr(filtered_text, edge_features, node_features, vec_dict,
                  part=None, edge_para=None, node_para=None, **kwargs):
    """
    edge feature
    word attraction rank
    filterted_text为空格连接的单词序列，edge_features和vecs为dict
    特征计算后append到edge_features中

    params: filtered_text, filtered normalized string
            edge_features, a edge:feature dict
            vec_dict,
    """
    # 词向量的格式不统一，要想办法处理
    def force(freq1, freq2, distance):
        return freq1 * freq2 / (distance * distance)

    def dice(freq1, freq2, edge_count):
        return 2 * edge_count / (freq1 * freq2)

    # 统计force和共现次数的总和，以便标准化
    if '+' in part:
        edge_force = {}
        edge_ctr = {}
        force_sum = 0
        count_sum = 0
        ctr_sum = 0
        for edge in edge_features:
            splited = filtered_text.split()
            freq1 = splited.count(edge[0])
            freq2 = splited.count(edge[1])

            default_vec = [1] * len(list(vec_dict.values())[0])
            vec1 = vec_dict.get(edge[0], default_vec)
            vec2 = vec_dict.get(edge[1], default_vec)
            distance = euc_distance(vec1, vec2)
            attraction_force = force(freq1, freq2, distance)
            edge_force[edge] = attraction_force
            force_sum += attraction_force
            count_sum += edge_features[edge][0]

            edge_gx = edge_features[edge][:3]
            ctr = sum([i * j for i, j in zip(edge_gx, edge_para)])
            edge_ctr[edge] = ctr
            ctr_sum += ctr

    for edge in edge_features:
        splited = filtered_text.split()
        freq1 = splited.count(edge[0])
        freq2 = splited.count(edge[1])

        # 读不到的词向量设为全1
        default_vec = [1] * len(list(vec_dict.values())[0])
        vec1 = vec_dict.get(edge[0], default_vec)
        vec2 = vec_dict.get(edge[1], default_vec)
        distance = euc_distance(vec1, vec2)

        if part == 'force*gx':
            edge_count = edge_features[edge][0]
            word_attr = force(freq1, freq2, distance) * edge_count
        elif part == 'force+gx':
            edge_count = edge_features[edge][0]
            part_weight = kwargs['part_weight']
            word_attr = part_weight * \
                edge_force[edge] / force_sum + \
                (1 - part_weight) * edge_count / count_sum
        elif part == 'force*ctr':
            edge_gx = edge_features[edge][:3]
            ctr = sum([i * j for i, j in zip(edge_gx, edge_para)])
            word_attr = force(freq1, freq2, distance) * ctr
        elif part == 'force+ctr':
            part_weight = kwargs['part_weight']
            word_attr = part_weight * \
                edge_force[edge] / force_sum + \
                (1 - part_weight) * edge_ctr[edge] / ctr_sum
        elif part == 'force*other':
            edge_gx = edge_features[edge][:3]
            edge_try = 1
            for i in edge_gx:
                edge_try *= i + 1
            word_attr = force(freq1, freq2, distance) * edge_try
            # word_attr = edge_try
        else:
            word_attr = force(freq1, freq2, distance) * \
                dice(freq1, freq2, edge_count)

        edge_features[edge].append(word_attr)

    return edge_features
