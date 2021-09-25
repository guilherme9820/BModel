from scipy.stats import circmean
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors


sns.set(style='ticks', context='talk')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['legend.title_fontsize'] = 24


def parse_model_specs(specs):
    num_points = specs[-1]
    model_name = '_'.join(specs[:-1])

    return model_name, num_points


def format_model_name(model_name):

    if isinstance(model_name, str):
        specs = model_name.split("_")
        return parse_model_specs(specs)
    else:
        specs = model_name.str.split("_")
        specs = specs.apply(parse_model_specs)
        return pd.DataFrame(specs.to_list(), columns=['model', 'num_points'])


def load_data(csv_path):
    dataframe = pd.read_csv(csv_path)

    # dataframe["angular_difference"] = dataframe["angular_difference"].apply(np.rad2deg)
    # dataframe["angular_difference"] = dataframe["angular_difference"].replace(0, 1e-7)
    # dataframe["angular_difference"] = dataframe["angular_difference"].apply(np.abs)
    # dataframe["angular_difference"] = dataframe["angular_difference"].apply(np.log10)

    specs = format_model_name(dataframe['model'])

    dataframe.drop(['model', 'min_angle'], axis=1, inplace=True)
    dataframe[specs.columns] = specs

    return dataframe


def build_figure(ax, dataframe):

    max_angle = dataframe['max_angle'][0]

    dataframe.drop(['max_angle', 'num_points'], axis=1, inplace=True)
    dataframe = dataframe.pivot_table(index=['test_point'], columns=['model'], values=['angular_difference'], aggfunc=circmean)
    dataframe = dataframe.stack().reset_index()

    g = sns.boxplot(ax=ax, data=dataframe, x="model", y="angular_difference")
    for box, color in zip(g.artists, sns.color_palette()):
        box.set_edgecolor(color)
        box.set_facecolor(mcolors.to_rgba(color, alpha=0.5))

    fliers = []
    for line in ax.lines:
        if line.get_marker() != 'None':
            fliers.append(line)

    for flier, color in zip(fliers, sns.color_palette()):
        flier.set_marker("o")
        flier.set_markerfacecolor("None")
        flier.set_markeredgecolor(color)
        flier.set_markersize(10)
        flier.set_markeredgewidth(2)

    ax.set_ylim(2e-4, 5e-1)
    ax.set_xticklabels(["B-dcm", "B-quat", "6D", "A"])
    ax.set_ylabel("Angular error (rad)", fontsize=24)
    ax.set_xlabel(f"{max_angle}°", fontsize=24)
    ax.set(yscale="log")
    ax.grid(True, which="minor", ls="--", c='gray')
    ax.set_aspect(1./ax.get_data_ratio())
    ax.tick_params(axis='both', which='major', labelsize=20)


dataframes = [load_data("max_angle_10.csv"),
              load_data("max_angle_100.csv"),
              load_data("max_angle_180.csv")]

fig, axes = plt.subplots(1, len(dataframes), figsize=(16, 5), sharey=True)

for i, (ax, dataframe) in enumerate(zip(axes, dataframes)):
    build_figure(ax, dataframe)
    if i != 0:
        ax.set_ylabel("")
        ax.legend().set_visible(False)

# plt.tight_layout()
plt.subplots_adjust(left=0.1, bottom=0, right=0.95, top=1.1, wspace=0.1, hspace=0)
plt.show()
