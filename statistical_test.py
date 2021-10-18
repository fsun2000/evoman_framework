import os
import numpy as np
import pandas as pd
import json
from scipy import stats
import csv

for algo in ['CMA-ES', 'NEAT']:
    if algo == 'CMA-ES':
        # Load CMA-ES results
        df_path = os.path.join('CMA-ES' + '27', 'df-gains-per-enemy.csv')
        df1 = pd.read_csv(df_path, index_col=0)

        df_path = os.path.join('CMA-ES' + '78', 'df-gains-per-enemy.csv')
        df2 = pd.read_csv(df_path, index_col=0)
    else:
        # Load NEAT results
        with open(os.path.join('neat' + '27', 'gains.json')) as f:
            dictdata = json.load(f)
        for run in dictdata:
            for enemy in dictdata[run]:
                p, e = dictdata[run][enemy]['player_energy'], dictdata[run][enemy]['enemy_energy']
                dictdata[run][enemy] = np.mean(np.array(p) - np.array(e))
        df1 = pd.DataFrame(dictdata).T
        df1 = df1.rename(columns=lambda s: s[-1], index=lambda s: s[-1])

        with open(os.path.join('neat' + '78', 'gains.json')) as f:
            dictdata = json.load(f)
        for run in dictdata:
            for enemy in dictdata[run]:
                p, e = dictdata[run][enemy]['player_energy'], dictdata[run][enemy]['enemy_energy']
                dictdata[run][enemy] = np.mean(np.array(p) - np.array(e))
        df2 = pd.DataFrame(dictdata).T
        df2 = df2.rename(columns=lambda s: s[-1], index=lambda s: s[-1])

    print(df1.mean(axis=1).mean())
    print(df2.mean(axis=1).mean())

    group27_data = df1.values.T
    group78_data = df2.values.T

    data = np.array([])
    for i in range(len(group27_data)):
        data = np.concatenate((data, group27_data[i]), axis=0)
        data = np.concatenate((data, group78_data[i]), axis=0)
    data = data.reshape(16, 10)

    with open('stat_test_groupwise_' + algo + '.csv', 'a', newline='', encoding='utf-8') as f:
        for i in range(0, 16, 2):
            print(i)
            # ttest = stats.ttest_ind(data[i], data[i+1], equal_var=False)
            wilcoxon = stats.wilcoxon(data[i], data[i+1])
            writer = csv.writer(f)
            writer.writerow(wilcoxon)
        f.close()

