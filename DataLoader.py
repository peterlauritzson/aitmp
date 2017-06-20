from sklearn import datasets
import numpy as np
from sklearn.datasets import fetch_mldata


class DataLoader:
    def __init__(self, list_of_data_tuples):
        self.list_of_data_tuples = list_of_data_tuples
        self.idx = -1

    def generator(self):
        while True:
            self.increase_idx()
            yield self.list_of_data_tuples[self.idx]

    def increase_idx(self):
        if self.idx < 0 or self.idx > self.list_of_data_tuples.shape[0]:
            self.idx = 0
            np.random.shuffle(self.list_of_data_tuples)
        else:
            self.idx += 1


def load_mnist():
    mnist = fetch_mldata('MNIST original', data_home="./tmp/")
    return np.array(list(zip(mnist.data, mnist.target)))


def load_digits():
    digits = datasets.load_digits()
    return np.array(list(zip(digits.images, digits.target)))


def load_iris():
    iris = datasets.load_iris()
    return np.array(list(zip(iris.data, iris.target)))


def sinus_data(data_size, low=-2*np.pi, high=2*np.pi):
    points = np.arange(start=low, stop=high, step=(high-low)/data_size)
    target = np.sin(points)
    return np.array(list(zip(points, target)))


def linear_random(data_size, low=-10, high=10):
    points = np.arange(start=low, stop=high, step=(high-low)/data_size)
    target = np.array([x+np.random.normal() for x in points])
    return np.array(list(zip(points, target)))


