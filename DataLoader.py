from sklearn import datasets
import numpy as np
from sklearn.datasets import fetch_mldata


class DataLoader:
    def __init__(self, list_of_data_tuples):
        self.list_of_data_tuples = list_of_data_tuples
        self.idx = -10

    def generator(self):
        while True:
            self.increase_idx()
            yield self.list_of_data_tuples[self.idx]

    def increase_idx(self):
        self.idx += 1
        if self.idx < 0 or self.idx >= self.list_of_data_tuples.shape[0]:
            self.idx = 0
            np.random.shuffle(self.list_of_data_tuples)


def create_train_val_test_data(data, val_prop, test_prop):
    n_vals = int(len(data) * val_prop)
    n_tests = int(len(data) * test_prop)
    np.random.shuffle(data)
    train = data[:-n_tests-n_vals]
    val = data[-n_tests-n_vals+1:-n_tests]
    test = data[-n_tests+1:]
    return train, val, test


def load_mnist():
    mnist = fetch_mldata('MNIST original', data_home="./tmp/")
    n_values = np.max(mnist.target) + 1
    target = np.eye(int(round(n_values)))[mnist.target.astype(np.int64)]
    return np.array(list(zip(mnist.data, target)))


def load_digits():
    digits = datasets.load_digits()
    return np.array(list(zip(digits.images, digits.target)))


def load_iris():
    iris = datasets.load_iris()
    n_values = np.max(iris.target)+1
    target = np.eye(n_values)[iris.target]
    return np.array(list(zip(iris.data, target)))


def sinus_data(data_size, low=-2*np.pi, high=2*np.pi):
    points = np.arange(start=low, stop=high, step=(high-low)/data_size)
    target = np.sin(points)
    return np.array(list(zip(points, target)))


def linear_random(data_size, low=-10, high=10):
    points = np.arange(start=low, stop=high, step=(high-low)/data_size)
    target = np.array([x+np.random.normal() for x in points])
    return np.array(list(zip(points, target)))


def linear_data(data_size, low=-10, high=10):
    points = np.arange(start=low, stop=high, step=(high-low)/data_size)
    return np.array(list(zip(points, points)))


