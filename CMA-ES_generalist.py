# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os
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

    # The cma module uses the numpy random number generator
    # np.random.seed(128)

    # The CMA-ES algorithm takes a population of one individual as argument
    # The centroid is set to a vector of 5.0 see http://www.lri.fr/~hansen/cmaes_inmatlab.html
    # for more details about the rastrigin and other tests for CMA-ES    
    strategy = cma.Strategy(centroid=list(np.zeros(N)), sigma=np.sqrt(2**2/12), 
                            lambda_=int(np.ceil(mu/lambda_ratio)), mu=mu)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("std", np.std)
    
    #logger = tools.EvolutionLogger(stats.functions.keys())
   
    # The CMA-ES algorithm converge with good probability with those settings
    pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=n_gens, stats=stats, halloffame=hof)
    
    print("Best individual fitness is {}".format(hof[0].fitness.values))
    return pop, logbook, hof

from pathlib import Path
if __name__ == "__main__":
    en = [2, 7, 8]
    for lambda_ratio in [0.3, 0.35, 0.4, 0.45, 0.5]:
        for i in range(5):
            np.random.seed(i)
            
            log_path = Path('deap_grid_search', 'CMA-ES_{}'.format(lambda_ratio), 'run-{}'.format(i))
            log_path.mkdir(parents=True, exist_ok=True)

            pop, logbook, hof = main(name=str(log_path), enemies=en, n_gens=15, lambda_ratio=lambda_ratio, mu=50, seed=i)

            best = hof.items[0]
            print("-- Best Individual = ", np.array(best))
            print("-- Best Fitness = ", best.fitness.values[0])

            # Export training results
            df_log = pd.DataFrame(logbook)
            results_path = os.path.join(log_path, 'stats.csv')
            # Remove previous experiment results
            if os.path.exists(results_path):
                os.remove(results_path)
            with open(results_path, 'w') as csv_file:
                df_log.to_csv(csv_file, index=False, line_terminator='\n') 

            # # Save best solution
            # with open("deap_grid_search/solutions/" + "solution_" + logbook_name +  ".pickle", 'wb') as pickle_file:
            #     pickle.dump(np.array(best), pickle_file)

