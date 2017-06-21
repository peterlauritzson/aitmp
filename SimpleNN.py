from random import random
import numpy as np
import math


def sigmoid(x):
    if x < -500:
        return 0
    if x > 500:
        return 1
    return 1 / (1 + math.exp(-x))


def linear(x):
    return x


class NeuralNetwork:
    def __init__(self, input_dim, layer_sizes, output_dim,
                 hidden_activation_function=sigmoid, output_activation_function=np.tanh):
        self.nodes = list()  # List of lists, each sublist contains all nodes at that layer
        self.nodes.append(list())
        for i in range(input_dim):
            node_name = "Input layer, node: " + str(i)
            self.nodes[0].append(Node(hidden_activation_function, node_name))
        for i_layer in range(len(layer_sizes)):
            self.nodes.append(list())
            for i_node in range(layer_sizes[i_layer]):
                node_name = "Hidden layer: " + str(i_layer) + ", node: " + str(i_node)
                node = Node(hidden_activation_function, node_name)
                for prev_nodes in self.nodes[i_layer]:
                    node.connect(prev_nodes)
                self.nodes[i_layer+1].append(node)
        self.nodes.append(list())
        for i in range(output_dim):
            node_name = "Output layer, node: " + str(i)
            node = Node(output_activation_function, node_name)
            for prev_nodes in self.nodes[-2]:
                node.connect(prev_nodes)
            self.nodes[-1].append(node)
        self.num_biases = -1

    def predict(self, input_data):
        if len(self.nodes[0]) == 1:
            self.nodes[0][0].value = input_data
        else:
            for data, node in zip(input_data, self.nodes[0]):
                node.value = data
        for i_layer in range(1, len(self.nodes)):
            for node in self.nodes[i_layer]:
                node.compute_value()
        result = list()
        for node in self.nodes[-1]:
            result.append(node.value)
        return np.array(result)

    def get_weights_and_biases(self):
        weights = list()
        biases = list()
        for i_layer in range(len(self.nodes)-1):
            for node in self.nodes[i_layer]:
                for child_node in self.nodes[i_layer+1]:
                    weights.append(node.get_weight(child_node))
            for node in self.nodes[i_layer+1]:
                biases.append(node.get_bias())
        self.num_biases = len(biases)
        return weights, biases

    def set_weights_and_biases(self, weights, biases):
        i_weight = 0
        i_bias = 0
        for i_layer in range(len(self.nodes)-1):
            for node in self.nodes[i_layer]:
                for child_node in self.nodes[i_layer+1]:
                    node.set_weight(child_node, weights[i_weight])
                    i_weight += 1
            for node in self.nodes[i_layer+1]:
                node.set_bias(biases[i_bias])
                i_bias += 1

    def set_all(self, x, n_biases=None):
        if n_biases is None:
            n_biases = self.num_biases
        self.set_weights_and_biases(x[:-n_biases], x[n_biases+1:])

    def get_node(self, layer, count):
        """

        :param layer: 0 for input, -1 for output, int for others
        :param count:
        :return:
        """
        return self.nodes[layer][count]


class Node:
    def __init__(self, activation_function, name):
        self.parents = list()
        self.activation_function = activation_function
        self.no_parent = True
        self.value = 0
        self.name = name
        self.weights = dict()
        self.bias = np.random.normal()

    def compute_value(self):
        self.value = 0
        for parent in self.parents:
            self.value += parent.value * parent.weights[self]
        self.value += self.bias
        self.value = self.activation_function(self.value)

    def connect(self, parent_node):
        self.parents.append(parent_node)
        self.no_parent = False

        parent_node.weights[self] = np.random.normal()

    def remove_connection_to_child(self, child):
        del self.weights[child]

    def update_value(self, value):
        self.value = value

    def get_weight(self, child):
        return self.weights[child]

    def get_bias(self):
        return self.bias

    def set_bias(self, bias):
        self.bias = bias

    def set_weight(self, child, new_weight):
        self.weights[child] = new_weight
