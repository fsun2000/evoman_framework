# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os
from pathlib import Path
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import copy

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

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
            player_controller=player_controller(10),
            enemies=enemies,
            multiplemode="yes",
            randomini='yes',
            savelogs='no',     # Save logs with DEAP instead
            playermode="ai",
            speed="fastest",
            enemymode="static",
            logs="off") # Turn off to disable logs in terminal
        Experiment.env.cons_multi = cons_multi

def fitness(ind):
    f,p,e,_ = Experiment.env.play(pcont=np.array(ind))

    g = gain(p,e)
    if g > Experiment.best_gain:
        Experiment.best_genome = copy.deepcopy(np.array(ind))
        Experiment.best_gain = g

    # return as list for sequence coercion in DEAP
    return [f]

# Problem size
N=265

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("evaluate", fitness)

def main(name, enemies, n_gens, lambda_ratio, mu, seed):
    np.random.seed(seed)
    Experiment.initialize(name, enemies)

    strategy = cma.Strategy(centroid=np.random.uniform(-1, 1, N), sigma=np.sqrt(2**2/12), 
                            lambda_=int(np.ceil(mu/lambda_ratio)), mu=mu)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("std", np.std)

    # The CMA-ES algorithm converge with good probability with those settings
    pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=n_gens, stats=stats, halloffame=hof)
    
    print("Best individual fitness is {}".format(hof[0].fitness.values))
    return pop, logbook, hof


if __name__ == "__main__":
    LAMBDA_RATIO = 0.3 # See parameter search results in best_CMA_params.py
    N_GENS = 30
    MU = 50
    en = [2,7]

    for i in range(10):
        np.random.seed(i)
        
        log_path = Path('CMA-ES' + ''.join([str(e) for e in en]), 'run-{}'.format(i))
        log_path.mkdir(parents=True, exist_ok=True)

        pop, logbook, hof = main(name=str(log_path), enemies=en, n_gens=N_GENS, lambda_ratio=LAMBDA_RATIO, mu=MU, seed=i)

        best = hof.items[0]
        print("-- Best Individual = ", np.array(best))
        print("-- Best Fitness = ", best.fitness.values[0])

        # Export training results
        df_log = pd.DataFrame(logbook)
        results_path = os.path.join(log_path, 'final-stats.csv')
        # Remove previous experiment results
        if os.path.exists(results_path):
            os.remove(results_path)
        with open(results_path, 'w') as csv_file:
            df_log.to_csv(csv_file, index=False, line_terminator='\n') 

        # Save best solution based on our gain
        gain_solution_path = os.path.join(log_path, 'gain-solution.npy')
        np.save(gain_solution_path, Experiment.best_genome)

        # Save best solution based on our fitness
        fitness_solution_path = os.path.join(log_path, 'fitness-solution.npy')
        np.save(fitness_solution_path, np.array(best))
            
