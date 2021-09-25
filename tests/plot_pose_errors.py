import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import circmean

pose_errors = pd.read_csv("pose_errors.csv")
pose_errors.drop(columns=['min_angle', 'max_angle'], inplace=True)

sns.set(style='ticks', context='talk')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['legend.title_fontsize'] = 24

plt.figure(figsize=(12, 12))
g = sns.lineplot(data=pose_errors, x="test_point", y="angular_difference", hue="model", palette="bright", estimator=circmean, ci=95)

labels = g.get_legend_handles_labels()[1]
handles = []
for label, color in zip(labels, sns.color_palette()):
    specs = label.split("_")
    num_points = specs[-1]
    parametrization = specs[-2]
    model_name = '_'.join(specs[:-2])
    if model_name == "bmodel":
        label = f"B-{parametrization} ({num_points})"
    elif model_name == "smooth_representation":
        label = f"A ({num_points})"
    else:
        label = f"6D ({num_points})"

    handles.append(mlines.Line2D([],
                                 [],
                                 color=color,
                                 marker='s',
                                 markersize=15,
                                 linestyle='',
                                 label=label))

g.legend(handles=handles,
         loc="upper right",
         prop={"size": 18},
         ncol=2)
g.grid(True, which="minor", ls="--", c='gray')
g.set(yscale="log")
g.set_ylabel("Angular error (rad)", fontsize=30)
g.set_xlabel("Consumed points", fontsize=30)
g.tick_params(axis='both', which='major', labelsize=30)
plt.tight_layout()
plt.show()
