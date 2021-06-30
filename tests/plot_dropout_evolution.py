import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dropout_evolution = pd.read_csv("dropout_evolution.csv")
dropout_evolution["var"] = dropout_evolution["var"].apply(np.rad2deg)
dropout_evolution["test_dropout"] = dropout_evolution["test_dropout"]
dropout_evolution["train_dropout"] = dropout_evolution["train_dropout"]

sns.set()
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['legend.title_fontsize'] = 24

plt.figure(figsize=(12, 12))
g = sns.lineplot(data=dropout_evolution, x="test_dropout", y="var", hue="train_dropout", palette="bright")
g.lines[0].set_linestyle("dashed")
g.lines[1].set_linestyle("dashed")
g.lines[2].set_linestyle("dashed")
g.legend(title="Training rate (%)", loc="upper left", prop={"size": 24})
g.set_ylabel("Variance (deg²)", fontsize=30)
g.set_xlabel("Testing rate (%)", fontsize=30)
g.tick_params(axis='both', which='major', labelsize=30)
plt.tight_layout()

plt.figure(figsize=(12, 12))
g2 = sns.lineplot(data=dropout_evolution, x="test_dropout", y="wahba_error", hue="train_dropout", palette="bright")
g2.legend(title="Training rate (%)", loc="upper left", prop={"size": 24})
g2.set_ylabel("Wahba's error", fontsize=30)
g2.set_xlabel("Testing rate (%)", fontsize=30)
g2.tick_params(axis='both', which='major', labelsize=30)

plt.tight_layout()
plt.show()
