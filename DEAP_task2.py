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

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


# choose this for not using visuals and thus making experiments faster
# headless = True
# if headless:
#     os.environ["SDL_VIDEODRIVER"] = "dummy"

PARAM_CONTROL = False
N_HIDDEN_NODES = 10

experiment_name = 'Task2_multi'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
                  enemies=[2,7,8], # Tune parameters on mixed group [2,7,8]
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(N_HIDDEN_NODES),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  randomini="yes")

def fitness(ind):
    f,p,e,t = env.play(pcont=np.array(ind))
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

# Length of one genome
IND_SIZE = (env.get_num_sensors()+1) * N_HIDDEN_NODES + (N_HIDDEN_NODES+1) * 5

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

# Store best individual ever lived
hof = tools.HallOfFame(1)

# Initialize (mu, lambda) model hyperparams
MU = 30
LAMBDA = 3 * MU
NGENS = 15

# Initial population
pop = toolbox.population(n=MU)

# Perform (mu, lambda) algorithm using crossover_prob=0.6 and mutate_prob = 0.3
pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, 
            cxpb=0.6, mutpb=0.3, ngen=NGENS, stats=stats, halloffame=hof, verbose=False)

# Export training results
df_log = pd.DataFrame(logbook)
with open("Task2_multi/logbook.csv", 'w') as csv_file:
    df_log.to_csv(csv_file, index=False) 

with open("Task2_multi/logbook.pickle", 'wb') as pickle_file:
    pickle.dump(logbook, pickle_file)

# Plot results, don't save plot.
gen = logbook.select("gen")
# fit_mins = logbook.chapters["fitness"].select("max")
# fit_avgs = logbook.chapters["fitness"].select("avg")
fit_mins = logbook.select("max")
fit_avgs = logbook.select("avg")
fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, fit_mins, "b-", label="Maximum Fitness")
line1 = ax1.plot(gen, fit_avgs, "b-", label="Average Fitness")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness", color="b")
plt.show()