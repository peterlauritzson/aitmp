import random
import math
import numpy as np
from FitnessFunctions import beale_function as fitness

DIM = 2
N_GENERATIONS = 1000
N_PARTICLES = 15
LOWER_BOUNDS = [-3, -3]
UPPER_BOUNDS = [3, 3]
W = 0.7
C1 = 2
C2 = 2


class Particle:
    def __init__(self, lower_bounds, upper_bounds, fitness_function, save_history=False, w=W, c1=C1, c2=C2):
        self.position = list()
        self.velocity = list()
        for lb, ub in zip(lower_bounds, upper_bounds):
            domain_size = abs(ub - lb)/10
            self.position.append(random.uniform(lb, ub))
            self.velocity.append(random.uniform(-domain_size, domain_size))

        self.prev_best = self.position
        self.swarm_best = self.prev_best
        self.fitness_function = fitness_function
        self.fitness = self.fitness_function(self.position)

        self.save_history = save_history
        self.history = list()

        self.w = w
        self.c1 = c1
        self.c2 = c2

    def update(self):
        rp = random.uniform(0, 1)
        rg = random.uniform(0, 1)
        self.velocity = [self.w * v + self.c1 * rp * (pb - x) + self.c2 * rg * (sb - x)
                         for x, v, pb, sb in zip(self.position, self.velocity, self.prev_best, self.swarm_best)]
        self.position = [x + v for x, v in zip(self.position, self.velocity)]
        self.fitness = self.fitness_function(self.position)
        if self.save_history:
            self.history.append(self.position)


def create_swarm(n_particles, lower_bounds, upper_bounds, fitness_function, save_history=False):
    swarm = list()
    for _ in range(n_particles):
        swarm.append(Particle(lower_bounds, upper_bounds, fitness_function, save_history=save_history))

    # Set the best known position in the swarm:
    swarm_best_value = math.inf
    swarm_best_position = []
    for particle in swarm:
        if particle.fitness < swarm_best_value:
            swarm_best_value = particle.fitness
            swarm_best_position = particle.position

    for particle in swarm:
        particle.swarm_best = swarm_best_position

    return swarm


def run_algorithm(save_particles=False):
    swarm = create_swarm(N_PARTICLES, LOWER_BOUNDS, UPPER_BOUNDS, fitness, save_history=save_particles)

    generation = 0
    swarm_best_value = math.inf
    swarm_best_position = []

    while generation < N_GENERATIONS:
        generation += 1
        for particle in swarm:
            particle.update()
            if particle.fitness < swarm_best_value:
                swarm_best_value = particle.fitness
                swarm_best_position = particle.position

        for particle in swarm:
            particle.swarm_best = swarm_best_position

        print("Gen: " + str(generation) + ", fitness: " + str(swarm_best_value))

    if save_particles:
        all_paths = list()
        for particle in swarm:
            all_paths.append(particle.history)
        return all_paths


run_algorithm()

