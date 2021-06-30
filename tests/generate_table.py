import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import dataframe_image as dfi


def get_min_indices(pivot_table):

    # For each target angle gets the entry with the lowest error mean
    min_indices = []
    for index in pivot_table.index:
        idx = pivot_table.loc[index][" "].argmin()
        min_indices.append(idx)

    return min_indices


def format_table(dataframe):

    dataframe = dataframe.rename(columns={"model": "Model",
                                          "dropout": "Dropout rate (%)",
                                          "case": "Case",
                                          "mean_wahba_error": " "})

    dataframe = pd.pivot_table(dataframe,
                               index=['Case'],
                               columns=['Model', 'Dropout rate (%)'],
                               values=[" "])

    return dataframe


def apply_style(dataframe, indices, style):

    # df.style.apply iterates over each row of the dataframe,
    # so indide the function 'highlight_min' we must retrieve the
    # correct index for each row. Hence, a generator holding the
    # indices was the best option.
    index_generator = (index for index in indices)

    def highlight_min(row, style):

        # Gets the correct index
        index = next(iter(index_generator))

        target = [False for _ in range(len(row))]

        target[index] = True

        # Only the row position that is set to true will have
        # the font-weight changed.
        return [style if v else '' for v in target]

    if isinstance(dataframe, pd.core.frame.DataFrame):
        return dataframe.style.apply(highlight_min, axis=1, **{'style': style})

    return dataframe.apply(highlight_min, axis=1, **{'style': style})


def build_table_image(dataframe, indices, image_path='df_styled.png'):

    # Applies serif font family to all values
    dataframe = dataframe.style.set_properties(**{'font-family': 'serif'})
    # Applies bold font to minimum values
    dataframe = apply_style(dataframe, indices, style='font-weight: bold')
    # Applies blue font color to minimum values
    dataframe = apply_style(dataframe, indices, style='color: blue')

    # Center-aligns the header texts and sets serif as font family
    dataframe = dataframe.set_table_styles(
        [{'selector': 'th',
          'props': [('text-align', 'center'),
                    ('font-family', 'serif')]}]
    )

    # Saves the dataframe as a .png file
    dfi.export(dataframe, image_path, fontsize=20)


def main(results):

    table = format_table(results)

    min_indices = get_min_indices(table)

    table = table.round(2)

    build_table_image(table.astype(str), min_indices, image_path="wahba_error4.pdf")


if __name__ == "__main__":

    results = pd.read_csv("uncertainties.csv")

    results["mean_wahba_error"] = results["mean_wahba_error"].apply(np.log10)

    main(results)

# pivot_2obs = pivot_2obs.style.apply(highlight_min, axis=1)

# dfi.export(pivot_2obs, 'df_styled.png')
# df.style.apply(, axis='index')

# results = results.groupby(['target_angle', 'obs'])

# min_row = results.get_group((15, 2)).min()
# min_row2 = results.get_group((15, 3)).min()

# index = np.where(results == min_row)

# print(pd.join([min_row, min_row2]))
# results = results.round(2)

# results["uncertainty"] = results["angle_mean"].astype(str) + "±" + results["angle_std"].astype(str)
# results = results.drop(columns=['angle_mean', "angle_std"])

# pivot_table = pd.pivot_table(results,
#                              index=['target_angle', 'obs'],
#                              columns=['model', 'dropout'],
#                              values=['uncertainty'],
#                              aggfunc=lambda x: ' '.join(x))

# pivot_table.to_csv("formated_results.csv")

# results["angle_std"] = results["angle_std"].apply(np.rad2deg)

# two_obs = results[results["obs"] == 2]

# # model_3obs = results[results["model"] == "3obs"]
# # model_4obs = results[results["model"] == "4obs"]
# # model_5obs = results[results["model"] == "5obs"]
# # model_6obs = results[results["model"] == "6obs"]
# # model_7obs = results[results["model"] == "7obs"]

# facet_kws = {"subplot_kws": dict(projection='polar'), "despine": False}


# sns.relplot(x="target_angle", y="angle_std", hue="model", size="dropout",
#             sizes=(40, 400), alpha=1., palette="colorblind",
#             height=6, data=two_obs, facet_kws=facet_kws)
# # ax.plot(model_3obs[""], stats, 'o', ms=2)
# # ax.fill(angles, stats, alpha=0.25)
# # ax.set_thetagrids(angles * 180/np.pi, labels)
# # ax.set_title([df.loc[386, "Name"]])
# # ax.grid(True)
# plt.show()
