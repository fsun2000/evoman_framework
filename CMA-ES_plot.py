"""
Plots the training fitness results for CMA-ES with uniformly initialized centroid.
Note that for both groups, run 7 had a bad initialization. CMA-ES was able to overcome this on
group [2, 7], but not on [7, 8], where it did not find a valid solution.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

csv_name = 'final-stats.csv'
plt.rcParams['font.size'] = '18'
x = range(30)
fig, ax = plt.subplots(figsize=(10,6))

for group in [[2, 7], [7, 8]]:
    mean_fit, max_fit = [], []
    for run in range(10):
        csv_path = str(Path('CMA-ES' + ''.join([str(e) for e in group]), 'run-{}'.format(run), csv_name))
        csv_data = pd.read_csv(csv_path, delimiter=',')
        mean_fit.append(csv_data['avg'])
        max_fit.append(csv_data['max'])

    means_avg = np.mean(mean_fit, axis=0)
    means_max = np.mean(max_fit, axis=0)
    stds_avg = np.std(mean_fit, axis=0)
    stds_max = np.std(max_fit, axis=0)

    ax.plot(x, means_max, '-', label=str(group) + ' max')
    ax.fill_between(x, means_max - stds_max, means_max + stds_max, alpha=0.15)
    ax.plot(x, means_avg, '-', label=str(group) + ' mean')
    ax.fill_between(x, means_avg - stds_avg, means_avg + stds_avg, alpha=0.15)

    ax.set_xlabel("Generation", fontsize=22)
    ax.set_ylabel("Fitness", fontsize=22)
ax.set_title("CMA-ES performance per group", fontsize=22)
ax.legend()
plt.tight_layout()
plt.show()
plt.savefig('CMA-ES27/training_fitness.png')
plt.savefig('CMA-ES78/training_fitness.png')    


