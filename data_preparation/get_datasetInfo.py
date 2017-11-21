import os
import csv
from re import match


def getFilenames(path):
    filenames = []
    for name in os.listdir(path):
        filenames.append(name)
    return filenames


def writeFilenames(path, filenames):
    with open(path, mode='w', encoding="utf-8") as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(filenames)


def readAfile(path):
    with open(path, mode='r', encoding='utf-8') as f:
        text = f.read()
    text = text.replace('\n', ' ')
    return text


def writeAfile(path, text):
    with open(path, mode='a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')


def istag(tag):
    return match(r'^[A-Z].+', tag)


def textFormat(text):
    words = [word for word in text.split()]
    context = []
    for word in words:
        wordAndtag = word.split('_')
        if istag(wordAndtag[-1]):
            context.append(wordAndtag[0].lower())
    context = ' '.join(context)
    return context


def datasetsInfo(datasets=['KDD', 'WWW']):
    #datasets = ['KDD', 'WWW']
    for dataset in datasets:
        abstractsPath = '../dataset/' + dataset + '/abstracts'
        filenamesPath = './data_temp/' + dataset + '/abstractsNames'
        abstractsText = './data_temp/' + dataset + '/abstracts.data'
        if os.path.exist(abstractsText):
            os.remove(abstractsText)

        # get and save abstracts'filenames
        abstractsNames = getFilenames(abstractsPath)
        writeFilenames(filenamesPath, abstractsNames)

        # get and save abstracts'text
        for abstractsName in abstractsNames:
            abstractsNamePath = abstractsPath + '/' + abstractsName
            text = readAfile(abstractsNamePath)
            text = textFormat(text)
            writeAfile(abstractsText, text)


if __name__ == "__main__":
    datasetsInfo()
