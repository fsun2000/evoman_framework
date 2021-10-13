from argparse import ArgumentParser
import copy
import json
from pathlib import Path
import os
import pickle
import sys

import numpy as np

# evoman
sys.path.insert(0, 'evoman')
os.environ["SDL_VIDEODRIVER"] = "dummy"
from controller import Controller
from environment import Environment

# neat
import neat


class NeatController(Controller):
    def sigmoid_activation(self, x):
        return 1. / (1. + np.exp(-x))

    def control(self, inputs, net):
        # Normalises the input using min-max scaling
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))

        output = np.array(net.activate(inputs))
        output = self.sigmoid_activation(output)
        return np.round(output)


# Overwrite cons_multi to just return raw values
def cons_multi(values):
    return values


# Use original function for fitness
def fitness(values):
    return values.mean() - values.std()


# Use gain according to doc
def gain(p, e):
    return p.sum() - e.sum()


# This is a singleton so that I don't have to use global variables
class Experiment:
    def initialize(name, enemies):
        Experiment.init_env(name, enemies)
        Experiment.best_genome = None
        Experiment.best_gain = -1000

    def init_env(name, enemies):
        Experiment.env = Environment(
            experiment_name=name,
            player_controller=NeatController(),
            enemies=enemies,
            multiplemode="yes",
            randomini='yes',
            savelogs='no',     # Save logs with NEAT instead
            playermode="ai",
            speed="fastest",
            enemymode="static")
        Experiment.env.cons_multi = cons_multi


def evaluate(genomes, config):
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        f, p, e, _ = Experiment.env.play(pcont=net)

        g = gain(p,e)
        if g > Experiment.best_gain:
            Experiment.best_genome = copy.deepcopy(genome)
            Experiment.best_gain = g
        genome.fitness = fitness(f)


def parse_args(args):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--name', type=str, help='Experiment name', default='generalist_neat')
    parser.add_argument('--config', type=Path, help='Path to config file', default=Path('config_neat'))
    parser.add_argument('--checkpoint', type=Path, help='Path to checkpoint to load')
    parser.add_argument('--ch-interval', type=int, help='Checkpoint interval per generation', default=1)
    parser.add_argument('--max-gen', type=int, help='Maximum number of generations', default=30)
    parser.add_argument('--enemies', type=int, nargs='+', help='Enemies to use', default=[2, 7, 8])
    return parser.parse_args(args)


def main(args):
    parsed_args = parse_args(args)
    # Load NEAT configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, parsed_args.config)
    all_gains = {}

    # Run 10 independent experiments
    for run in range(10):
        log_path = Path(parsed_args.name, 'run-{}'.format(run))
        log_path.mkdir(parents=True, exist_ok=True)
        Experiment.initialize(str(log_path), parsed_args.enemies)

        pop = neat.Population(config)

        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.Checkpointer(parsed_args.ch_interval,
                            filename_prefix=log_path.joinpath('checkpoint-')))

        # Run experiment
        pop.run(evaluate, parsed_args.max_gen)
        # Save mean and max fitness by generation
        stats.save_genome_fitness(filename=str(log_path.joinpath('stats.csv')))

        # Save "best" solution
        with Path(log_path, 'best-genome').open('wb') as outfile:
            pickle.dump(Experiment.best_genome, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        # Get gains for "best" solution
        Experiment.init_env('all', [1,2,3,4,5,6,7,8])
        net = neat.nn.FeedForwardNetwork.create(Experiment.best_genome, config)
        all_gains['run-{}'.format(run)] = {}
        for _ in range(5):
            _, player, enemy, _ = Experiment.env.play(pcont=net)
            for i, (p, e) in enumerate(zip(player, enemy)):
                if 'enemy-{}'.format(i+1) not in all_gains['run-{}'.format(run)]:
                    all_gains['run-{}'.format(run)]['enemy-{}'.format(i+1)] = {}
                    all_gains['run-{}'.format(run)]['enemy-{}'.format(i+1)]['player_energy'] = []
                    all_gains['run-{}'.format(run)]['enemy-{}'.format(i+1)]['enemy_energy'] = []
                all_gains['run-{}'.format(run)]['enemy-{}'.format(i+1)]['player_energy'].append(int(p))
                all_gains['run-{}'.format(run)]['enemy-{}'.format(i+1)]['enemy_energy'].append(int(e))

    # Save gains
    with Path(parsed_args.name, 'gains.json').open('w') as outfile:
        json.dump(all_gains, outfile, indent=4)


if __name__ == '__main__':
    main(sys.argv[1:])
