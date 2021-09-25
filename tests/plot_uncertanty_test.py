import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def plot_var(axes, data, font_size=12):

    data["case"] = data["case"].astype('int32')
    data["observation"] = data["observation"].apply(lambda x: f"{x:.0f} obs")

    sns.barplot(ax=axes[0], data=data, x="case", y="var_x", hue="observation")
    sns.barplot(ax=axes[1], data=data, x="case", y="var_y", hue="observation")
    sns.barplot(ax=axes[2], data=data, x="case", y="var_z", hue="observation")

    axes[0].set_xlabel("")
    axes[0].legend(prop={"size": font_size}, ncol=2)
    axes[0].set_ylabel(r'$var(\vec{\mathbf{u}}_x)\;(deg^{2})$', fontsize=font_size)
    axes[0].tick_params(axis='both', which='major', labelsize=font_size)
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    axes[1].set_xlabel("")
    axes[1].set_ylabel(r'$var(\vec{\mathbf{u}}_y)\;(deg^{2})$', fontsize=font_size)
    axes[1].tick_params(axis='both', which='major', labelsize=font_size)
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    axes[2].set_xlabel("Test case", fontsize=font_size)
    axes[2].set_ylabel(r'$var(\vec{\mathbf{u}}_z)\;(deg^{2})$', fontsize=font_size)
    axes[2].tick_params(axis='both', which='major', labelsize=font_size)
    axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    axes[1].legend().set_visible(False)
    axes[2].legend().set_visible(False)

    plt.tight_layout()


sns.set(style='ticks', context='talk')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'normal'

results = pd.read_csv("uncertainty_test.csv")

results["var_x"] = results["var_x"].apply(np.rad2deg)
results["var_y"] = results["var_y"].apply(np.rad2deg)
results["var_z"] = results["var_z"].apply(np.rad2deg)

grouped = results.groupby("dropout")

drop10 = grouped.get_group(10)
drop15 = grouped.get_group(15)
drop20 = grouped.get_group(20)

fig1, axes1 = plt.subplots(3, 1, figsize=(18, 12))
# fig1.suptitle(r"Dropout rate of $10\%$")
plot_var(axes1, drop10, 26)

fig2, axes2 = plt.subplots(3, 1, figsize=(18, 12))
# fig2.suptitle(r"Dropout rate of $15\%$")
plot_var(axes2, drop15, 26)

fig3, axes3 = plt.subplots(3, 1, figsize=(18, 12))
# fig3.suptitle(r"Dropout rate of $20\%$")
plot_var(axes3, drop20, 26)

plt.show()
