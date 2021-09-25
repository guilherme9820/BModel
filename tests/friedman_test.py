import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import dataframe_image as dfi
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from Orange.evaluation import compute_CD
from scipy.stats import rankdata
from Orange.evaluation import graph_ranks


# sns.set()
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'normal'
# plt.rcParams['legend.title_fontsize'] = 60


def format_table(dataframe):

    dataframe = dataframe.rename(columns={"observation": "Model",
                                          "dropout": "Dropout rate (%)",
                                          "case": "Case",
                                          "mean_wahba_error": " "})

    dataframe = pd.pivot_table(dataframe,
                               index=['Case'],
                               columns=['Model', "Dropout rate (%)"],
                               values=[" "])

    return dataframe


def friedman_test(data, names):

    friedmanstats = friedmanchisquare(*data)

    print(f"Chi-Square: {friedmanstats[0]}; p-value: {friedmanstats[1]}")

    data = np.array(data).T

    if friedmanstats[1] < 0.05:

        posthoc = sp.posthoc_siegel_friedman(data, p_adjust='fdr_bh')

        plt.figure(figsize=(12, 8))
        heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.82, 0.35, 0.02, 0.2]}
        heatmap, cbar = sp.sign_plot(posthoc, **heatmap_args)
        heatmap.set_xticklabels(names)
        heatmap.set_yticklabels(names)
        heatmap.tick_params(labelsize=24)
        cbar.ax.tick_params(labelsize=24)
        plt.tight_layout()

        ranks = np.apply_along_axis(rankdata, axis=1, arr=data)

        avgranks = np.mean(ranks, axis=0)

        cd = compute_CD(avgranks, data.shape[0], test="nemenyi")

        print(f"Critical Difference: {cd}")

        graph_ranks(avgranks, names, cd=cd, width=6, textspace=1.5)

        plt.show()


def main(results):

    table = format_table(results)

    cols = [table[(' ', 3, 10)].values,
            table[(' ', 4, 10)].values,
            table[(' ', 5, 10)].values,
            table[(' ', 6, 10)].values,
            table[(' ', 7, 10)].values]

    friedman_test(cols, names=['3obs', '4obs', '5obs', '6obs', '7obs'])

    cols = [table[(' ', 3, 15)].values,
            table[(' ', 4, 15)].values,
            table[(' ', 5, 15)].values,
            table[(' ', 6, 15)].values,
            table[(' ', 7, 15)].values]

    friedman_test(cols, names=['3obs', '4obs', '5obs', '6obs', '7obs'])

    cols = [table[(' ', 3, 20)].values,
            table[(' ', 4, 20)].values,
            table[(' ', 5, 20)].values,
            table[(' ', 6, 20)].values,
            table[(' ', 7, 20)].values]

    friedman_test(cols, names=['3obs', '4obs', '5obs', '6obs', '7obs'])


if __name__ == "__main__":

    results = pd.read_csv("uncertainty_test.csv")

    # results["mean_wahba_error"] = results["mean_wahba_error"].apply(np.log10)

    main(results)
