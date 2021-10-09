import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

for en in [6]:
    label_names = ['point CNA', 'uniform CNA']
    csv_names = ['point-crossover-results-100.csv', 'improved-results-100.csv']
    plt.rcParams['font.size'] = '18'

    fig, ax = plt.subplots(figsize=(10,6))
    
    for i, dir in enumerate(['CNA_point_crossover_scores','CNA_final_scores']):
        data = []
        for run in range(10):
            path = os.path.join(dir, 'enemy-{}'.format(en), 'run-{}'.format(run), csv_names[i])

            data.append(np.genfromtxt(path, delimiter=','))

        data = np.array(data)

        # (mean_average, mean_max)
        means = np.mean(data, axis=0)
        # (std_average, std_max)
        stds = np.std(data, axis=0)

        means_avg = means[:, 0]
        stds_avg = stds[:, 0]

        means_max = means[:, 1]
        stds_max = stds[:, 1]

        x = range(100)
        # Plot means, maxes and save figures
        ax.plot(x, means_max, '-', label=label_names[i] + ' max')
        ax.fill_between(x, means_max - stds_max, means_max + stds_max, alpha=0.2)
        ax.plot(x, means_avg, '-', label=label_names[i] + ' mean')
        ax.fill_between(x, means_avg - stds_avg, means_avg + stds_avg, alpha=0.2)
    ax.set_xlabel("Generation", fontsize=22)
    ax.set_ylabel("Fitness", fontsize=22)
    ax.set_title("Enemy 6", fontsize=22)
    ax.legend()
    plt.tight_layout()
    plt.savefig('uniform-point-enemy-{}.png'.format(en))










