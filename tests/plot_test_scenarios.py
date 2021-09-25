import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
paper_rc = {'lines.markersize': 15}
sns.set(style='ticks', context='talk', rc=paper_rc)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'normal'

comparison = pd.read_csv("test_scenarios.csv")
comparison["algorithm"].iloc[np.where(comparison['algorithm'] == "nn")] = "neural network"
comparison["algorithm"].iloc[np.where(comparison['algorithm'] == "q_method")] = "q-method"

plt.figure(figsize=(18, 12))
g = sns.barplot(data=comparison,
                x="test_case",
                y="loss",
                hue="algorithm")
g.set_yscale("log")
g.legend(loc="upper center", prop={"size": 24}, ncol=2)
g.set_xlabel('Test Case', fontsize=28)
g.set_ylabel("Wahba's error (log)", fontsize=28)
g.tick_params(axis='both', which='major', labelsize=28)
plt.tight_layout()

plt.show()
