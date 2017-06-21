from scipy.optimize import basinhopping

from DataLoader import DataLoader, linear_random, sinus_data, linear_data, load_iris, load_mnist, \
    create_train_val_test_data
from FitnessFunctions import predictor_fitness, network_opt
from SimpleNN import NeuralNetwork, linear
from sklearn.metrics import log_loss, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


def example_one_to_one_nn_optimization():
    simple_nn = NeuralNetwork(1, [3, 3, 3], 1, output_activation_function=linear)
    #data = linear_data(10000)
    data = sinus_data(100)
    data_gen = DataLoader(data)
    neural_predictor = lambda nn: predictor_fitness(nn.predict, data_gen.generator(), loss_function=mean_squared_error)

    x0w, x0b = simple_nn.get_weights_and_biases()
    print("Number of weights:", len(x0w))
    print("Number of biases:", len(x0b))
    x0 = x0w
    x0.extend(x0b)
    fitness = lambda ind: network_opt(ind, simple_nn, neural_predictor)

    # OPTIMIZE HERE
    best_ind = [1, 1/2.0, 1/2.0, 1, 10, 10, -15]
    print("Best fitness is:", fitness(best_ind))
    print(simple_nn.get_weights_and_biases())
    #

    plt.plot([x[0] for x in data], [x[1] for x in data], 'b*')
    simple_nn.set_all(best_ind)
    plt.plot([x[0] for x in data], [simple_nn.predict(x[0]) for x in data], 'r*')
    plt.show()


def example_many_to_one_hot_nn_optimization():
    #data = load_iris()
    data = load_mnist()
    train, val, test = create_train_val_test_data(data, 0.25, 0.10)
    print(data[0][0].shape[0])
    print(data[0][1].shape[0])

    # VERIFICATION PLOT OF MNIST
    pixels = data[0][0].reshape((28, 28))

    # Plot
    plt.imshow(pixels, cmap='gray')
    plt.show()
    #

    data_gen = DataLoader(train)
    neural_predictor = lambda nn: predictor_fitness(nn.predict, data_gen.generator())

    simple_nn = NeuralNetwork(data[0][0].shape[0], [3, 3, 3], data[0][1].shape[0])
    x0w, x0b = simple_nn.get_weights_and_biases()
    print("Number of weights:", len(x0w))
    print("Number of biases:", len(x0b))
    x0 = x0w
    x0.extend(x0b)
    fitness = lambda ind: network_opt(ind, simple_nn, neural_predictor)

    # OPTIMIZE HERE
    best_ind = x0
    print("Best fitness is:", fitness(best_ind))
    print(simple_nn.get_weights_and_biases())
    #

    simple_nn.set_all(best_ind)
    data_gen = DataLoader(test)
    print("Fitness on test data:", predictor_fitness(simple_nn.predict, data_gen.generator(), batch_size=test.shape[0]))


#example_one_to_one_nn_optimization()
example_many_to_one_hot_nn_optimization()
