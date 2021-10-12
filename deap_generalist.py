""" 
Run basic experiment first (do grid-search)
Figure out compatibility with Island model / species preservation / CMA-ES
Then run some experiments for these
"""

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os
import array
import random
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import multiprocessing
import copy
import os

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

PARAM_CONTROL = True
N_HIDDEN_NODES = 10

# Overwrite cons_multi to just return mean without subtracting std.
def cons_multi(values):
    return values.mean()

# Use gain according to doc. Fitness function is kept the same.
def gain(p, e):
    return p - e


# This is a singleton so that I don't have to use global variables
class Experiment:
    def initialize(name, enemies):
        Experiment.init_env(name, enemies)
        Experiment.best_genome = None
        Experiment.best_gain = -301

    def init_env(name, enemies):
        Experiment.env = Environment(
            experiment_name=name,
            player_controller=player_controller(N_HIDDEN_NODES),
            enemies=enemies,
            multiplemode="yes",
            randomini='yes',
            savelogs='no',     # Save logs with DEAP instead
            playermode="ai",
            speed="fastest",
            enemymode="static",
            logs="on") # Turn off to disable logs in terminal
        Experiment.env.cons_multi = cons_multi

def fitness(ind):
    f,p,e,_ = Experiment.env.play(pcont=np.array(ind))

    g = gain(p,e)
    if g > Experiment.best_gain:
        Experiment.best_genome = copy.deepcopy(np.array(ind))
        Experiment.best_gain = g

    # return as list for sequence coercion in DEAP
    return [f]


if PARAM_CONTROL == True:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None)
    creator.create("Strategy", array.array, typecode="d")
else:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximise the fitness
    creator.create("Individual", list, fitness=creator.FitnessMax)


# Toolbox to store registered functions
toolbox = base.Toolbox()

# Length of one genome for the competition
IND_SIZE = 265 # (Experiment.env.get_num_sensors()+1) * N_HIDDEN_NODES + (N_HIDDEN_NODES+1) * 5

# Function used to randomly initialize one genome and its mutation strategy per allele
def generateES(ind_cls, strg_cls, size):
    ind = ind_cls(random.uniform(-1, 1) for _ in range(size))
    ind.strategy = strg_cls(np.random.normal(0, 1) for _ in range(size))
    return ind

# Generate a new individual
if PARAM_CONTROL == True:
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy, IND_SIZE)
else:
    toolbox.register("new_ind", np.random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                    toolbox.new_ind, n=IND_SIZE)

# Generate a population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Add evaluation function
toolbox.register("evaluate", fitness)

# Add remaining crossover, mutation, selection mechanisms
if PARAM_CONTROL == True:
    toolbox.register("mate", tools.cxESBlend, alpha=0.1)
    toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.3)
else:
    toolbox.register("mate", tools.cxBlend, alpha=0.1)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Save statistics of the run
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)


# Optimization algorithm
def main(experiment_name, enemies):
    Experiment.initialize(experiment_name, enemies)

    # Store best individual ever lived
    hof = tools.HallOfFame(1)

    # Initialize (mu, lambda) model hyperparams
    # For parameter search use: ngens=15, mu=30
    NGENS = 15
    MU = 30
    LAMBDA = 3 * MU

    # Grid-search these?
    CXPB = 0.6
    MUTPB = 0.3

    # Initial population
    pop = toolbox.population(n=MU)

    # Perform (mu, lambda) algorithm using crossover_prob=0.6 and mutate_prob = 0.3 (old parameters are 1.0 & 0.2)
    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, 
                cxpb=CXPB, mutpb=MUTPB, ngen=NGENS, stats=stats, halloffame=hof, verbose=True)
    
    return pop, logbook, hof


if __name__ == '__main__':
    # Uncomment to use all cores at once
    # It takes 137 s for first 2 generations
    # pool = multiprocessing.Pool() 
    # toolbox.register('map', pool.map) 
    
    # without multiprocessing, it takes 248 s for first 2 gens
    tic = timer()
    pop, logbook, hof = main(experiment_name='Temp', enemies=[2,7,8]) 
    print("Final runtime: ", timer() - tic)

    best = hof.items[0]
    print("-- Best Individual = ", np.array(best.strategy))
    print("-- Best Fitness = ", best.fitness.values[0])

    logbook_name = 'temp'
    # Save best solution
    with open("deap_grid_search/solutions/" + "solution_" + logbook_name +  ".pickle", 'wb') as pickle_file:
        pickle.dump(np.array(best.strategy), pickle_file)

    # Export training results
    df_log = pd.DataFrame(logbook)
    with open("deap_grid_search/scores/" + logbook_name + ".csv", 'w') as csv_file:
        df_log.to_csv(csv_file, index=False, line_terminator='\n') 




    # Code to plot fitness (doesn't save img)
    """
    gen = logbook.select("gen")
    fit_mins = logbook.select("max")
    fit_avgs = logbook.select("avg")
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Maximum Fitness")
    line1 = ax1.plot(gen, fit_avgs, "b-", label="Average Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    plt.show()
    """