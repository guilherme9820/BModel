import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sns.set(style='ticks', context='talk')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'normal'

results = pd.read_csv("angular_difference.csv")
results["angular_difference"] = results["angular_difference"].apply(np.rad2deg)
results["observations"] = results["observations"].apply(lambda x: f"{x} obs")

results = results.groupby(by="dropout")
drop10 = results.get_group(10)
drop15 = results.get_group(15)
drop20 = results.get_group(20)

plt.figure(figsize=(18, 12))
g = sns.boxplot(data=drop10, x="case", y="angular_difference", hue="observations")
g.legend(loc="upper left", prop={"size": 24})
g.set_xlabel('Test case', fontsize=30)
g.set_ylabel("Angular difference (deg)", fontsize=30)
g.tick_params(axis='both', which='major', labelsize=30)
plt.tight_layout()

plt.figure(figsize=(18, 12))
g = sns.boxplot(data=drop15, x="case", y="angular_difference", hue="observations")
g.legend(loc="upper left", prop={"size": 24})
g.set_xlabel('Test case', fontsize=30)
g.set_ylabel("Angular difference (deg)", fontsize=30)
g.tick_params(axis='both', which='major', labelsize=30)
plt.tight_layout()

plt.figure(figsize=(18, 12))
g = sns.boxplot(data=drop20, x="case", y="angular_difference", hue="observations")
g.legend(loc="upper left", prop={"size": 24})
g.set_xlabel('Test case', fontsize=30)
g.set_ylabel("Angular difference (deg)", fontsize=30)
g.tick_params(axis='both', which='major', labelsize=30)
plt.tight_layout()

plt.show()
