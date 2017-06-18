from random import Random, random


def linear(x):
    return x


class NeuralNetwork:
    def __init__(self):
        self.input_nodes = dict()

    def create_input_node(self, activation_function, name):
        self.input_nodes[name] = Node(activation_function, name=name)



class Node:
    def __init__(self, activation_function, name=""):
        self.parents = list()
        self.activation_function = activation_function
        self.no_parent = True
        self.value = 0
        self.name = name
        self.weights = dict()
        self.bias = random.gauss(0, 1)

    def compute_value(self):
        self.value = 0
        for parent in self.parents:
            self.value += parent.value * parent.weights[self]
        self.value += self.bias
        self.value = self.activation_function(self.value)

    def connect(self, parent_node):
        self.parents.append(parent_node)
        self.no_parent = False

        parent_node.weights[self] = random.gauss(0, 1)

    def update_value(self, value):
        self.value = value

    def update_weight(self, parent, new_weight):
        parent.weights[self] = new_weight
