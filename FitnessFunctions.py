import math
import numpy


class DataGenerator:
    def __init__(self, data_set, label_set):
        pass

    def generator_function(self):
        pass

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


def predictor_fitness(predictor, data_generator, batch_size=1000):
    fitness = 0
    for _ in range(batch_size):
        input_data, correct_output = data_generator.__next__()
        output = predictor(input_data)
        fitness += numpy.linalg.norm(correct_output - output)**2
    return fitness

