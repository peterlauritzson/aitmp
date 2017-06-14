import random
import math
from FitnessFunctions import all_ones as fitness


def create_new_individual(length):
    individual = list()
    for x in range(0,length):
        individual.append(random.randint(0, 1))
    return individual


def select_high_fitness_index(pop_len):
    proportional_sum_index = random.randrange(0, pop_len*(pop_len-1)/2)
    # idx*(idx-1) = 2*proportional_sum_index
    # idx2 - idx = 2c <-> idx2 - idx + 1/4 = 2c + 1/4 <->
    # (idx-1/2)^2 = 2c + 1/4 <-> idx = 1/2 + sqrt(2c+1/4)
    idx = math.ceil(1/2 + math.sqrt(2*proportional_sum_index+1/4))
    if idx > pop_len or idx < 0:
        raise AssertionError
    #print("index: "+str(pop_len - idx)+", size: "+str(pop_len))
    return pop_len - idx


def mutate(individual, number_of_bits):
    new_individual = list(individual)
    for x in range(0, number_of_bits):
        index = random.randrange(0, len(individual))
        new_individual[index] = (new_individual[index] + 1) % 2
    return new_individual





def purge_bottom(sorted_population, new_length):
    #return sorted_population[0:new_length]

    # # The below returns unique elements in the population
    pop = list()
    for i in range(0, len(sorted_population)-1):
        if sorted_population[i][1] != sorted_population[i+1][1]:
            pop.append(sorted_population[i])
    if len(pop) > new_length:
        return pop[0:new_length]
    else:
        for _ in range(len(pop), POP_SIZE):
            #pop.append(mutate(pop[select_high_fitness_index(len(pop))], NUMBER_OF_BITS))
            pop.append(create_new_individual(IND_LEN))
        return pop


def create_children(parent1, parent2):
    split = random.randrange(0, len(parent1))
    child1 = parent1[0:split]
    child1.extend(parent2[split:])
    child2 = parent2[0:split]
    child2.extend(parent1[split:])
    return child1, child2


GENERATIONS = 100
POP_SIZE = 100
IND_LEN = 100
NEW_CHILDREN = int(POP_SIZE / 2)
NEW_MUTATE = int(POP_SIZE / 5)
NUMBER_OF_BITS = int(IND_LEN/20)


def run_algorithm():
    population = list()
    for x in range(0,POP_SIZE):
        population.append( create_new_individual(IND_LEN) )

    count = 0
    while count < GENERATIONS:
        count += 1
        # fitness & sort
        sorted_list = list()
        for ind in population:
            sorted_list.append( (fitness( ind ), ind) )
        sorted_list.sort()
        sorted_list.reverse()

        print("Gen: " + str(count) + ", fitness: "+str(sorted_list[0][0]))

        # create_children
        new_children = list()
        for idx in range(0, int(NEW_CHILDREN/2)):
            kids = create_children(sorted_list[idx][1], sorted_list[random.randrange(0, POP_SIZE)][1])
            #ind1 = select_high_fitness_index(POP_SIZE)
            #ind2 = select_high_fitness_index(POP_SIZE)
            #kids = create_children(sorted_list[ind1][1], sorted_list[ind2][1])
            new_children.append(kids[0])
            new_children.append(kids[1])

        # mutate
        new_mutated = list()
        for _ in range(0, NEW_MUTATE):
            #index = select_high_fitness_index(POP_SIZE)
            index = random.randrange(0, len(population))
            new_mutated.append(mutate(population[index], NUMBER_OF_BITS))

        # purge current population
        new_generation = purge_bottom(sorted_list, len(population) - len(new_children) - len(new_mutated))

        # concat results
        population = [x[1] for x in new_generation]
        population.extend(new_children)
        population.extend(new_mutated)


run_algorithm()
