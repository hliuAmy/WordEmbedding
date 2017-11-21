# from abstracts.data get words'tfidf value by file
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


def filenames(path):
    with open(path, mode='r', encoding='utf-8') as f:
        filenames = f.read().split(',')
    return filenames


def fileTextList(path):
    with open(path, mode='r', encoding='utf-8') as f:
        filesText = [text for text in f.read().split(
            '\n') if len(text.strip()) != 0]
    return filesText


def tfidf(datasets=['KDD', 'WWW']):
    for dataset in datasets:
        # data preparation for tfidf train
        filelistPath = './data_temp/' + dataset + '/abstractsNames'
        filesTextPath = './data_temp/' + dataset + '/abstracts.data'
        filelist = filenames(filelistPath)
        filesText = fileTextList(filesTextPath)
        # path for result by dataset
        tfidfPath = './data_temp/' + dataset + '/tfidfByfile/'

        # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
        # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        tfidf = transformer.fit_transform(vectorizer.fit_transform(filesText))
        word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
        weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        # 将结果写入文件
        print(len(filelist))
        print(len(filesText))
        for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
            print("-------这里输出第", i, "类文本的词语tf-idf权重------")
            print(filelist[i])
            path = tfidfPath + filelist[i]
            with open(path, 'w') as file_pointer:
                for j in range(len(word)):
                    if weight[i][j] != 0:
                        file_pointer.write('%s %s\n' % (word[j], weight[i][j]))
            file_pointer.close()


def get_word_tfidf_Byfile(dataset, filename, nodelist):
    dataPath = './data_temp/' + dataset + '/tfidfByfile/' + filename
    with open(dataPath, mode='r', encoding='utf-8') as f:
        wordAndtfidf = f.readlines()
    word_tfidf = {}
    for line in wordAndtfidf:
        word, tfidf = line.split()
        word_tfidf[word] = tfidf
    node_tfidf = {}
    minTfidf = min(word_tfidf.values())
    for node in nodelist:
        if node in word_tfidf.keys():
            node_tfidf[node] = word_tfidf[node]
        else:
            node_tfidf[node] = minTfidf
    return node_tfidf


if __name__ == "__main__":
    tfidf()
