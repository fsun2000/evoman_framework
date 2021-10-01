import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Enemy 2, 6, 8
CNA_data = [[81.2,80.0,48.4,8.0,70.8,61.6,62.8,43.2,56.4,80.0], 
            [-16.19999999999994,-15.279999999999884,-70.0,-37.59999999999994,-30.67999999999994,-16.479999999999944,9.600000000000136,-9.359999999999895,-24.479999999999944,0.20000000000015455],
            [70.24000000000015,-12.759999999999973,33.88000000000022,19.520000000000227,23.40000000000012,77.92000000000016,13.320000000000098,75.16000000000017,-82.0,25.480000000000246]]

NEAT_data = [[87.2, 29.6, 79.6, 33.2, 74.8, 34.0, 40.0, 80.4, 59.6, 13.6],
            [37.84000000000028,-10.0,-14.879999999999935,-42.0,60.76000000000032,-2.159999999999877,-35.359999999999935,44.5600000000003,85.36000000000013,-50.0],
            [10.160000000000036,65.80000000000015,-22.759999999999973,65.9200000000001,29.920000000000037,-1.0399999999999523,-50.0,52.480000000000224,43.16000000000006,68.92000000000017]]

def merge_lists(l1, l2):
    res = []
    for i in range(len(l1)):
        res.append(l1[i])
        res.append(l2[i])
    return res
x = merge_lists(CNA_data, NEAT_data)

df = pd.DataFrame(x, index=['CNA_2', 'NEAT_2', 'CNA_6', 'NEAT_6', 'CNA_8', 'NEAT_8'])

fig = plt.figure(figsize=(12,5))
sns.set(font_scale=2.2)
ax=sns.boxplot(x="value", y="variable",data=pd.melt(df.T))
ax = sns.swarmplot(x="value", y="variable", data=pd.melt(df.T), color=".25")

ax.set_xlabel('Individual Gain',fontsize=28)
ax.set_ylabel('Algorithm', fontsize=28)
fig.tight_layout()

plt.savefig("Gains_boxplot.png")
plt.show()
