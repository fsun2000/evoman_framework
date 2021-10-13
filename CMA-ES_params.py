import pandas as pd
from pathlib import Path

maf_results = {}
mbf_results = {}

csv_name = 'stats.csv'
for lambda_ratio in [0.3, 0.35, 0.4, 0.45, 0.5]:
    all_filenames = [str(Path('CMA-ES_grid_search', 'CMA-ES_{}'.format(lambda_ratio), 'run-{}'.format(i), csv_name))
                     for i in range(5)]
    combined_csv_data = pd.concat([pd.read_csv(f, delimiter=',').tail(1) for f in all_filenames])
    maf_results[lambda_ratio] = combined_csv_data.avg.mean()
    mbf_results[lambda_ratio] = combined_csv_data['max'].mean()

print("Lambda ratio resulting in highest mean average fitness:")
print(max(maf_results, key=maf_results.get), maf_results[max(maf_results, key=maf_results.get)])
print("Lambda ratio resulting in highest mean best fitness:")
print(max(mbf_results, key=mbf_results.get), mbf_results[max(mbf_results, key=mbf_results.get)])
