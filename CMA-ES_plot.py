"""
Plot boxplots that compare the gains between the 4 configurations
[2, 7] vs [7, 8]
CMA-ES vs NEAT
"""
#  imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os
import csv
import seaborn as sns
import json

def plot_training_curve():
    csv_name = 'final-stats.csv'
    neat_name = 'stats.csv'
    plt.rcParams['font.size'] = '18'
    x = range(30)
    
    for group in [[2, 7], [7, 8]]:
        fig, ax = plt.subplots(figsize=(10,6))
        mean_fit, max_fit = [], []
        neat_mean_fit, neat_max_fit = [], []
        for run in range(10):
            csv_path = str(Path('CMA-ES' + ''.join([str(e) for e in group]), 'run-{}'.format(run), csv_name))
            csv_data = pd.read_csv(csv_path, delimiter=',')
            mean_fit.append(csv_data['avg'])
            max_fit.append(csv_data['max'])
            
            neat_path = str(Path('neat' + ''.join([str(e) for e in group]), 'run-{}'.format(run), neat_name))
            neat_data = pd.read_csv(neat_path, sep=' ',header=None)
            neat_mean_fit.append(neat_data.iloc[:, 1])
            neat_max_fit.append(neat_data.iloc[:, 0])

        means_avg = np.mean(mean_fit, axis=0)
        means_max = np.mean(max_fit, axis=0)
        stds_avg = np.std(mean_fit, axis=0)
        stds_max = np.std(max_fit, axis=0)

        neat_means_avg = np.mean(neat_mean_fit, axis=0)
        neat_means_max = np.mean(neat_max_fit, axis=0)
        neat_stds_avg = np.std(neat_mean_fit, axis=0)
        neat_stds_max = np.std(neat_max_fit, axis=0)

        ax.plot(x, means_max, '-', label= 'CMA-ES max')
        ax.fill_between(x, means_max - stds_max, means_max + stds_max, alpha=0.15)
        ax.plot(x, means_avg, '-', label= 'CMA-ES mean')
        ax.fill_between(x, means_avg - stds_avg, means_avg + stds_avg, alpha=0.15)

        ax.plot(x, neat_means_max, '-', label='NEAT max')
        ax.fill_between(x, neat_means_max - neat_stds_max, neat_means_max + neat_stds_max, alpha=0.15)
        ax.plot(x, neat_means_avg, '-', label='NEAT mean')
        ax.fill_between(x, neat_means_avg - neat_stds_avg, neat_means_avg + neat_stds_avg, alpha=0.15)

        ax.set_xlabel("Generation", fontsize=22)
        ax.set_ylabel("Fitness", fontsize=22)

        ax.set_title("Performance on group {}".format(group), fontsize=22)
        ax.legend()
        plt.tight_layout()
        group_string = ''.join([str(e) for e in group])
        plt.savefig('CMA-ES{}/training_fitness{}.png'.format(group_string, group_string), bbox_inches='tight')
        plt.show()

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# This is a singleton so that I don't have to use global variables
class Experiment:
    def initialize(enemies):
        Experiment.init_env(enemies)

    def init_env(enemies):
        Experiment.env = Environment(
            experiment_name="CMA-ES_plot",
            player_controller=player_controller(10),
            enemies=enemies,
            multiplemode="no",
            randomini='yes',
            savelogs='no',     # Save logs with DEAP instead
            playermode="ai",
            speed="fastest",
            enemymode="static",
            logs="on") # Turn off to disable logs in terminal

def measure_gain(solution):
    f,p,e,_ = Experiment.env.play(pcont=solution)
    return p - e

def get_best_run_energy(run=8, group=[2,7]):
    sol_name = 'gain-solution.npy'
    sol_path = str(Path('CMA-ES' + ''.join([str(e) for e in group]), 'run-{}'.format(run), sol_name))
    sol = np.load(sol_path)
    e_player = []
    e_enemy = []
    for en in range(1, 9):
        Experiment.initialize([en])
        p_sum, e_sum = 0, 0
        for i in range(5):
            f,p,e,_ = Experiment.env.play(pcont=sol)
            p_sum += p
            e_sum += e
        e_player.append(p_sum/5)
        e_enemy.append(e_sum/5)
    df = pd.DataFrame(np.array([e_player, e_enemy]))
    print(df)
    df.to_csv(str(Path('CMA-ES' + ''.join([str(e) for e in group]), 'best-run-energy.csv')), 
              header=list(range(1, 9)), index=False)


def create_boxplots():
    # Will contain mean avg gains for both groups per enemy
    for group in [[2, 7], [7, 8]]:
        # gains_per_enemy = []
        # for en in range(1, 9):
        #     Experiment.initialize([en])

        #     avg_gains = [] # 10 datapoints with the mean gains (5 runs)
        #     for run in range(10):#(10)
        #         gains = []
        #         for _ in range(5):
        #             sol_name = 'gain-solution.npy'
        #             sol_path = str(Path('CMA-ES' + ''.join([str(e) for e in group]), 'run-{}'.format(run), sol_name))
        #             sol = np.load(sol_path)
        #             gains.append(measure_gain(sol))
        #         mean_gain = np.mean(gains)
        #         avg_gains.append(mean_gain)
        #     gains_per_enemy.append(avg_gains)

        # print(gains_per_enemy)
        # filepath = os.path.join('CMA-ES' + ''.join([str(e) for e in group]), 'gains-per-enemy.csv')
        # with open(filepath, 'a', newline='', encoding='utf-8') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(gains_per_enemy)
        #     f.close()

        # Save CMA-ES results
        # df_path = os.path.join('CMA-ES' + ''.join([str(e) for e in group]), 'df-gains-per-enemy.csv')
        # df = pd.DataFrame(gains_per_enemy, index=['1', '2', '3', '4', '5', '6', '7', '8'])
        # with open(df_path, 'w') as csv_file:
        #     df.T.to_csv(csv_file) 

        # # Load NEAT results
        # with open(os.path.join('neat' + ''.join([str(e) for e in group]), 'gains.json')) as f:
        #     dict = json.load(f)
        # for run in dict:
        #     for enemy in dict[run]:
        #         p, e = dict[run][enemy]['player_energy'], dict[run][enemy]['enemy_energy']
        #         dict[run][enemy] = np.mean(np.array(p) - np.array(e))
        # df = pd.DataFrame(dict).T
        # df = df.rename(columns=lambda s: s[-1], index=lambda s: s[-1])

        for model_name in ['CMA-ES', 'neat']:
            if model_name == 'CMA-ES':
                # Load CMA-ES results
                df_path = os.path.join(model_name + ''.join([str(e) for e in group]), 'df-gains-per-enemy.csv')
                df = pd.read_csv(df_path, index_col=0)
            else:
                with open(os.path.join(model_name + ''.join([str(e) for e in group]), 'gains.json')) as f:
                    dict = json.load(f)
                for run in dict:
                    for enemy in dict[run]:
                        p, e = dict[run][enemy]['player_energy'], dict[run][enemy]['enemy_energy']
                        dict[run][enemy] = np.mean(np.array(p) - np.array(e))
                df = pd.DataFrame(dict).T
                df = df.rename(columns=lambda s: s[-1], index=lambda s: s[-1])
            # print(df.mean(axis=1))

            # To do: add mean marker, decrease marker size or use stripplot?
            fig = plt.figure(figsize=(8,5))
            sns.set(font_scale=2.2)
            ax = sns.boxplot(x="variable", y="value",data=pd.melt(df), showmeans=True, 
                            meanprops={"marker":"o",
                                        "markerfacecolor":"white", 
                                        "markeredgecolor":"black",
                                        "markersize":"5"})
            ax = sns.swarmplot(x="variable", y="value", data=pd.melt(df), color=".25")
            ax.set_title('Group {}, {}'.format(group, model_name.upper()), fontsize=25)
            ax.set_xlabel('Enemy', fontsize=22)
            ax.set_ylabel('Gain', fontsize=22)
            ax.set_ylim([-100,100])
            fig.tight_layout()
            config_name = model_name + ''.join([str(e) for e in group])
            img_path = os.path.join(config_name, config_name + '_boxplot.png')

            plt.savefig(img_path, bbox_inches='tight')
            plt.show()

if __name__ == "__main__":
    plot_training_curve()
    create_boxplots()

    # get_best_run_energy(run=8, group=[2,7])

"""
Best run CMA-ES [2,7]
0   -49.175
1   -22.405
2   -34.805
3   -33.130
4   -33.665
5   -43.315
6   -35.470
7   -16.820
8   -15.890  <---
9   -33.450
Run 8 mean gains
  -66.0, 40.8,-32.0,-70.0,  53.2,-90.0, 66.88,-30.0
Best run stats after 5 plays:
   0     1     2     3     4     5      6     7
 0.0  59.6   0.0   0.0  54.4   0.0  52.88   0.0 = player energy
68.0   2.0  38.0  66.0   0.0  90.0   4.00  30.0 = enemy  energy

Best run CMA-ES [7,8]
0   -20.645 <-
1   -27.725
2   -25.895
3   -30.430
4   -36.580
5   -27.485
6   -31.435
7   -36.945
8   -27.830
9   -31.345
"""

