import numpy as np
import csv
import os
import networkx


class Try:

    def __init__(self):
        self.A = 1

    def add(self, x):
        self.A = self.A + x
        # print(self.A)


class myTry:
    def __init__(self):
        self.A = 0

    def add(self, x):
        self.A = self.A + x
        # print(self.A)


if __name__ == '__main__':
    A = ['a', 'b']
    a = np.random.random_integers(8)
    print(np.random.random())
