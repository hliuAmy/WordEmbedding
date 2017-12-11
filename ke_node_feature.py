# ke_node_feature
# tfidf and sum(1/position)
from ke_preprocess import normalized_token, read_file, filter_text
import csv
import os
from collections import Iterable


def output(path, dict_word):
    with open(path, mode='w', encoding='utf-8') as f:
        csvWriter = csv.writer(f)
        for word, features in dict_word.items():
            row = []
            row.append(word)
            if isinstance(features, Iterable):
                row.extend(features)
            else:
                row.append(features)
            csvWriter.writerow(row)


def read_features(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.readlines()
    node_features = {}
    for line in text:
        line = line.strip('\n').split(',')
        if len(line) != 0:
            node_features[line[0]] = [float(x) for x in line[1:]]
    return node_features


def add_feature(path, word_feature):
    if os.path.exists(path):
        node_features = read_features(path)
        for key in node_features:
            if key in word_feature:
                node_features[key].append(word_feature[key])
            else:
                node_features[key].append(0)
        output(path, node_features)
    else:
        output(path, word_feature)


def add_position():
    filenames = read_file('../dataset/KDD_filelist').split(',')
    filepath = '../dataset/KDD/abstracts/'
    savePath = '../dataset/KDD/node_features/'
    for filename in filenames:
        filtered_text = filter_text(read_file(filepath + filename))
        filtered_text = filtered_text.split()
        word_position = {}
        for i in range(len(filtered_text)):
            word = filtered_text[i]
            if word not in word_position.keys():
                word_position[word] = float(1) / (i + 1)
            else:
                word_position[word] = word_position[word] + float(1) / (i + 1)
        add_feature(savePath + filename, word_position)


def add_tfidf():
    filenames = read_file('../dataset/KDD_filelist').split(',')
    tfidf_path = '../data_preparation/data_temp/KDD/tfidfByfile/'
    savePath = '../dataset/KDD/node_features/'
    for filename in filenames:
        filename = filename.strip()
        filepath = tfidf_path + filename
        file = read_file(filepath).split('\n')
        word_tfidf = {}
        for line in file:
            if len(line) != 0:
                word, tfidf = line.split(' ')
                word_tfidf[word] = float(tfidf)
        add_feature(savePath + filename, word_tfidf)


def del_feature():
    filenames = read_file('../dataset/KDD_filelist').split(',')
    tfidf_path = '../data_preparation/data_temp/KDD/tfidfByfile/'
    savePath = '../dataset/KDD/node_features/'
    for filename in filenames:
        filename = filename.strip()
        filepath = tfidf_path + filename
        file = read_file(filepath).split('\n')
        word_tfidf = []
        for line in file:
            if len(line) != 0:
                a, b = line.split(',')
                word_tfidf.append(a)
        with open(filepath, 'w', encoding='utf-8') as f:
            for line in word_tfidf:
                f.write(line + '\n')


if __name__ == '__main__':
    # add_position()
    # add_tfidf()
    print(os.path.exists('../data_preparation/data_temp/KDD/tfidfByfile/998449'))
