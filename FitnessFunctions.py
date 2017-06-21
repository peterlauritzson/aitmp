import math
from sklearn.metrics import log_loss, mean_squared_error
import numpy as np


def all_ones(individual):
    fit = 0
    for number in individual:
        if number == 1:
            fit += 1
    return fit


#Beale's function, global minimum at (3,0.5)
def beale_function(individual):
    x = individual[0]
    y = individual[1]
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2


# Rastrigin function, global minimum at 0^n
def rastrigin_function(individual, A=10):
    fitness = A*len(individual)
    for gene in individual:
        fitness += (gene**2 - A * math.cos(2*math.pi*gene))
    return fitness


def predictor_fitness(predictor, data_generator, loss_function=log_loss, batch_size=100):
    correct_outputs = list()
    outputs = list()
    for _ in range(batch_size):
        input_data, correct_output = data_generator.__next__()
        correct_outputs.append(correct_output)
        output = predictor(input_data)
        outputs.append(output)
    return loss_function(np.array(correct_outputs), np.array(outputs))


def network_opt(individual, neural_net, fitness):
    neural_net.set_all(individual)
    return fitness(neural_net)
