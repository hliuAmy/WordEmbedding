import os
import csv


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


if __name__ == "__main__":
    datasets = ['KDD', 'WWW']
    for dataset in datasets:
        abstractsPath = '../dataset/' + dataset + '/abstracts'
        filenamesPath = './data_temp/' + dataset + '/abstractsNames'
        abstractsText = './data_temp/' + dataset + '/abstracts.data'

        # get and save abstracts'filenames
        abstractsNames = getFilenames(abstractsPath)
        writeFilenames(filenamesPath, abstractsNames)

        for abstractsName in abstractsNames:
            abstractsNamePath = abstractsPath + '/' + abstractsName
            text = readAfile(abstractsNamePath)
            writeAfile(abstractsText, text)
