import math
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from sklearn.metrics import (
    d2_absolute_error_score,
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
)

from deep_metabolitics.utils.trainer_fcnn import q_error_function
from deep_metabolitics.utils.utils import load_pickle


def calc_q_error(y_true, y_pred):
    return np.median(q_error_function(y_true=y_true, y_pred=y_pred))


def root_mse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))


metric_func_mapping = {
    "R2": r2_score,
    # "Explained Variance": explained_variance_score,
    # "D2 Absolute Error": d2_absolute_error_score,
    # "Max Error": max_error,
    "Mean Absolute Error": mean_absolute_error,
    "Root Mean Squared Error": root_mse,
    # "Mean Absolute Percentage Error": mean_absolute_percentage_error,
    # "Mean Gamma Deviance": mean_gamma_deviance,
    # "Mean Poisson Deviance": mean_poisson_deviance,
    # "Mean Squared Error": mean_squared_error,
    # "Mean Squared Log Error": mean_squared_log_error,
    # "Median Absolute Error": median_absolute_error,
    # "Q Error Function": calc_q_error,
}


def plot_scores_foreach_dataset_foreach_methods(
    legend_list,
    data_sets,
    categories,
    metric_name,
    stat_for_pathway_metrics_q,
    experience_name,
    metrics_folder,
):
    # Kategoriler
    # categories = ["Breast", "ccRCC3", "ccRCC4", "Coad", "Pdac", "Prostate"]

    # Renk skalaları (her veri seti için farklı renk)
    # cmaps = ["Blues", "Reds", "Greens", "Purples", "Oranges", "Greys"]

    # vivid_cmaps = [
    #     "viridis",
    #     "plasma",
    #     "magma",
    #     "inferno",
    #     "cividis",
    #     "coolwarm",
    #     "rainbow",
    #     "Spectral",
    #     "hsv",
    #     "turbo",
    # ]

    # # Rastgele 100 colormap seç
    # cmaps = np.random.choice(vivid_cmaps, size=100, replace=True)
    all_cmaps = list(plt.colormaps())

    # İlk 100 colormap'i seç (gerekirse rastgele seçilebilir)
    cmaps = random.sample(all_cmaps, 100)
    colors = [
        [cm.get_cmap(cmap, 20)(15) for i in range(len(categories))] for cmap in cmaps
    ]

    # Plot oluşturma
    fig, ax = plt.subplots(figsize=(12, 8))

    # Her kategori ve veri seti için box plot
    offsets = np.linspace(
        -0.3, 0.3, len(data_sets)
    )  # Farklı veri setleri için yatay ofset
    # width = 0.2  # Kutuların genişliği

    for data_idx, (data, color_set) in enumerate(zip(data_sets, colors)):
        for i, category in enumerate(categories):
            ax.boxplot(
                data[i],
                positions=[
                    i + offsets[data_idx]
                ],  # Kategori pozisyonuna ofset ekleniyor
                # widths=width,
                patch_artist=True,
                boxprops=dict(facecolor=color_set[i], alpha=0.8, color=color_set[i]),
                medianprops=dict(color="black"),
                whiskerprops=dict(
                    color=color_set[i], linewidth=1.5
                ),  # Whisker renkleri
                capprops=dict(color=color_set[i], linewidth=1.5),  # Cap renkleri
            )

    # X ekseni ayarları
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_xlabel("Cancer Datasets")
    ax.set_ylabel(metric_name)
    ax.set_title(
        f"Performance of Different Metabolic Diff Score Prediction Models in Various Cancer Datasets ({metric_name})"
    )
    for x in [0.5, 1.5, 2.5, 3.5, 4.5]:
        ax.axvline(x=x, color="gray", linestyle="--", linewidth=0.6)

    # Legend ekleme (her veri setinin F kategorisindeki rengine göre)
    legend_patches = [
        plt.Line2D(
            [0], [0], color=colors[data_idx][-1], lw=10, label=legend_list[data_idx]
        )
        for data_idx in range(len(data_sets))
    ]
    ax.legend(handles=legend_patches, title="Solutions", loc="lower left")
    # Grafiği 300 DPI olarak kaydetme
    fig.savefig(
        metrics_folder
        / f"{experience_name}_{metric_name}_{stat_for_pathway_metrics_q}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_one_metric_for_all_methods_all_datasets(
    metric_fnames,
    dataset_mapping,
    all_approaches_data,
    metric_name,
    stat_for_pathway_metrics_q,
    experience_name,
    metrics_folder,
):
    legend_names = list(metric_fnames.values())
    categories = dataset_mapping.values()
    plot_scores_foreach_dataset_foreach_methods(
        legend_names,
        [list(v.values()) for v in all_approaches_data.values()],
        categories,
        metric_name=metric_name,
        stat_for_pathway_metrics_q=stat_for_pathway_metrics_q,
        experience_name=experience_name,
        metrics_folder=metrics_folder,
    )


def get_all_pathway_metrics_for_one_fold(pathway_metrics, metric_method):
    all_pathway_metrics = []
    for pathway_name, pathway_values in pathway_metrics.items():
        prediction_values = pathway_values["predicted"]
        true_values = pathway_values["actual"]
        metric_value = metric_method(y_true=true_values, y_pred=prediction_values)
        all_pathway_metrics.append(metric_value)
    return all_pathway_metrics


def get_all_pathway_metrics_for_all_folds(
    metrics, dataset_key, metric_method, stat_for_pathway_metrics_q="mean"
):
    all_fold_metrics = []
    for fold in metrics.keys():
        pathway_metrics = metrics[fold][dataset_key]["pathway_metrics"]
        one_fold_pathway_metrics = get_all_pathway_metrics_for_one_fold(
            pathway_metrics, metric_method
        )
        if stat_for_pathway_metrics_q == "mean":
            all_fold_metrics.append(np.mean(one_fold_pathway_metrics))
        else:
            all_fold_metrics.append(
                np.percentile(one_fold_pathway_metrics, stat_for_pathway_metrics_q)
            )
    return all_fold_metrics


def return_fold_metrics_for_all_approaches(
    metric_fnames,
    dataset_mapping,
    metric_method,
    metrics_folder,
    stat_for_pathway_metrics_q="mean",
):
    all_approaches_data = {}

    for fname in metric_fnames:
        metrics = load_pickle(metrics_folder / fname)
        all_approaches_data[metric_fnames[fname]] = {}

        # Her dataset için r2_50 değerlerini topla
        for dataset_key in dataset_mapping.keys():
            all_fold_metrics = get_all_pathway_metrics_for_all_folds(
                metrics,
                dataset_key,
                metric_method,
                stat_for_pathway_metrics_q=stat_for_pathway_metrics_q,
            )
            all_approaches_data[metric_fnames[fname]][dataset_key] = all_fold_metrics

    return all_approaches_data


def make_df_for_metric_overall(
    all_approaches_data,
    dataset_mapping,
    stat_for_pathway_metrics_q,
    metric_name,
    experience_name,
    metrics_folder,
):
    df = pd.DataFrame(
        [
            (
                method,
                dataset_mapping[dataset],
                np.mean(values),
            )
            for method, datasets in all_approaches_data.items()
            for dataset, values in datasets.items()
        ],
        columns=[
            "Method",
            "Dataset",
            metric_name,
        ],
    )
    df_pivot = df.pivot(
        index="Dataset",
        columns="Method",
        values=[
            metric_name,
        ],
    )
    fpath = (
        metrics_folder
        / f"{experience_name}_{metric_name}_{stat_for_pathway_metrics_q}.csv"
    )
    df_pivot.to_csv(fpath)
    return df_pivot


def make_overall_metrics_for_all_metrics(
    metric_fnames,
    dataset_mapping,
    experience_name,
    metrics_folder,
    stat_for_pathway_metrics_q=50,
):
    df_list = []
    # Dataset isimleri

    for metric_name, metric_func in metric_func_mapping.items():
        try:
            all_approaches_data = return_fold_metrics_for_all_approaches(
                metric_fnames=metric_fnames,
                dataset_mapping=dataset_mapping,
                metrics_folder=metrics_folder,
                metric_method=metric_func,
                stat_for_pathway_metrics_q=stat_for_pathway_metrics_q,
            )
            plot_one_metric_for_all_methods_all_datasets(
                metric_fnames=metric_fnames,
                dataset_mapping=dataset_mapping,
                all_approaches_data=all_approaches_data,
                metric_name=metric_name,
                stat_for_pathway_metrics_q=stat_for_pathway_metrics_q,
                experience_name=experience_name,
                metrics_folder=metrics_folder,
            )
            df = make_df_for_metric_overall(
                all_approaches_data=all_approaches_data,
                dataset_mapping=dataset_mapping,
                stat_for_pathway_metrics_q=stat_for_pathway_metrics_q,
                metric_name=metric_name,
                experience_name=experience_name,
                metrics_folder=metrics_folder,
            )
            df_list.append(df)
        except Exception as e:
            print(e)
            continue
    if df_list:
        overall_df = pd.concat(df_list, axis=1)
        overall_df.to_csv(metrics_folder / f"{experience_name}_overall_metrics.csv")
        return overall_df
    else:
        print("there is no df")


# def plot_pathway_predictions(
#     pathway_actual_arrays,
#     pathway_predicted_arrays,
#     source,
#     save_dir=None,
#     cols=3,
#     figsize=(20, 125),
# ):
#     """
#     Plot actual vs predicted values for all pathways in a single figure

#     Args:
#         pathway_metrics: List of dictionaries containing metrics for each pathway
#         save_dir: Directory to save the plots (optional)
#         cols: Number of columns in the subplot grid
#         figsize: Figure size as (width, height)
#     """
#     import math

#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     pathway_names = list(pathway_actual_arrays.keys())
#     n_pathways = len(pathway_names)
#     rows = math.ceil(n_pathways / cols)

#     fig = plt.figure(figsize=figsize)

#     r2_list = []
#     mse_list = []
#     mae_list = []
#     mape_list = []
#     evs_list = []
#     qe_list = []

#     for idx, pathway_name in enumerate(pathway_names, 1):
#         actual = pathway_actual_arrays[pathway_name]
#         predicted = pathway_predicted_arrays[pathway_name]

#         r2 = r2_score(y_true=actual, y_pred=predicted)
#         mse = mean_squared_error(y_true=actual, y_pred=predicted)
#         mae = mean_absolute_error(y_true=actual, y_pred=predicted)
#         mape = mean_absolute_percentage_error(y_true=actual, y_pred=predicted)
#         evs = explained_variance_score(y_true=actual, y_pred=predicted)
#         qe = calc_q_error(y_true=actual, y_pred=predicted)
#         r2_list.append(r2)
#         mse_list.append(mse)
#         mae_list.append(mae)
#         mape_list.append(mape)
#         evs_list.append(evs)
#         qe_list.append(qe)
#         ax = fig.add_subplot(rows, cols, idx)

#         # Create scatter plot
#         sns.scatterplot(x=actual, y=predicted, alpha=0.5, ax=ax)

#         # Add diagonal line (perfect prediction line)
#         min_val = min(actual.min(), predicted.min())
#         max_val = max(actual.max(), predicted.max())
#         ax.plot(
#             [min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction"
#         )

#         # Add horizontal and vertical lines at y=0 and x=0
#         ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
#         ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

#         # Add metrics text
#         text = f"R² = {r2:.3f}\nRMSE = {mse:.3f}\nMAE = {mae:.3f}\nMAPE = {mape:.3f}\nEV = {evs:.3f}\nQE = {qe:.3f}"
#         ax.text(
#             0.05,
#             0.95,
#             text,
#             transform=ax.transAxes,
#             verticalalignment="top",
#             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
#         )

#         # Labels and title
#         ax.set_title(f"{pathway_name}", fontsize=10)
#         ax.set_xlabel("Actual Values")
#         ax.set_ylabel("Predicted Values")
#     print(np.mean(r2_list), np.percentile(r2_list, 50))
#     print(np.mean(mse_list), np.percentile(mse_list, 50))
#     print(np.mean(mae_list), np.percentile(mae_list, 50))
#     print(np.mean(mape_list), np.percentile(mape_list, 50))
#     print(np.mean(evs_list), np.percentile(evs_list, 50))
#     print(np.mean(qe_list), np.percentile(qe_list, 50))

#     # Adjust layout to prevent overlap
#     plt.tight_layout()

#     # Save or show plot
#     if save_dir:
#         plt.savefig(
#             f"{save_dir}/{source}_all_pathways_predictions.png",
#             dpi=300,
#             bbox_inches="tight",
#         )
#         # plt.close()
#         plt.show()
#     else:
#         plt.show()


def plot_pathway_predictions(
    pathway_actual_arrays,
    pathway_predicted_arrays,
    source,
    save_dir=None,
    cols=3,
    figsize=(20, 125),
):
    """
    Plot actual vs predicted values for all pathways, saving both individual plots
    and a combined figure
    """
    import math

    import matplotlib.pyplot as plt
    import seaborn as sns

    pathway_names = list(pathway_actual_arrays.keys())
    n_pathways = len(pathway_names)
    rows = math.ceil(n_pathways / cols)

    # Lists to store metrics
    r2_list, mse_list, mae_list = [], [], []
    mape_list, evs_list, qe_list = [], [], []

    # Create combined figure
    fig = plt.figure(figsize=figsize)

    for idx, pathway_name in enumerate(pathway_names, 1):
        # actual = np.histogram(pathway_actual_arrays[pathway_name])[1]
        # predicted = np.histogram(pathway_predicted_arrays[pathway_name])

        actual = pathway_actual_arrays[pathway_name]
        predicted = pathway_predicted_arrays[pathway_name]

        # actual = np.histogram(pathway_actual_arrays[pathway_name])[1]
        # predicted = np.histogram(pathway_predicted_arrays[pathway_name])[1]

        # Calculate metrics
        r2 = r2_score(y_true=actual, y_pred=predicted)
        mse = mean_squared_error(y_true=actual, y_pred=predicted)
        mae = mean_absolute_error(y_true=actual, y_pred=predicted)
        mape = mean_absolute_percentage_error(y_true=actual, y_pred=predicted)
        evs = explained_variance_score(y_true=actual, y_pred=predicted)
        qe = calc_q_error(y_true=actual, y_pred=predicted)

        # Append metrics to lists
        r2_list.append(r2)
        mse_list.append(mse)
        mae_list.append(mae)
        mape_list.append(mape)
        evs_list.append(evs)
        qe_list.append(qe)

        # Create and save individual plot
        if save_dir:
            fig_single = plt.figure(figsize=(10, 8))
            ax_single = fig_single.add_subplot(111)
            _create_pathway_plot(
                ax_single, actual, predicted, pathway_name, r2, mse, mae, mape, evs, qe
            )
            plt.tight_layout()
            plt.savefig(
                f"{save_dir}/{source}_{pathway_name.replace('/', '_')}_prediction.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_single)

        # Add to combined plot
        ax = fig.add_subplot(rows, cols, idx)
        _create_pathway_plot(
            ax, actual, predicted, pathway_name, r2, mse, mae, mape, evs, qe
        )

    # Print average metrics
    print(f"Average metrics (mean, median):")
    print(f"R2: {np.mean(r2_list):.3f}, {np.percentile(r2_list, 50):.3f}")
    print(f"MSE: {np.mean(mse_list):.3f}, {np.percentile(mse_list, 50):.3f}")
    print(f"MAE: {np.mean(mae_list):.3f}, {np.percentile(mae_list, 50):.3f}")
    print(f"MAPE: {np.mean(mape_list):.3f}, {np.percentile(mape_list, 50):.3f}")
    print(f"EVS: {np.mean(evs_list):.3f}, {np.percentile(evs_list, 50):.3f}")
    print(f"QE: {np.mean(qe_list):.3f}, {np.percentile(qe_list, 50):.3f}")

    # Save or show combined plot
    plt.tight_layout()
    if save_dir:
        plt.savefig(
            f"{save_dir}/{source}_all_pathways_predictions.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


def _create_pathway_plot(
    ax, actual, predicted, pathway_name, r2, mse, mae, mape, evs, qe
):
    """Helper function to create a single pathway plot"""
    # Create scatter plot
    sns.scatterplot(x=actual, y=predicted, alpha=0.5, ax=ax)

    # Add diagonal line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")

    # Add reference lines
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

    # Add metrics text
    text = f"R² = {r2:.3f}\nRMSE = {mse:.3f}\nMAE = {mae:.3f}\nMAPE = {mape:.3f}\nEV = {evs:.3f}\nQE = {qe:.3f}"
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Labels and title
    ax.set_title(f"{pathway_name}", fontsize=10)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")


def get_all_pathway_actual_predicted(test_metrics, ds_name):
    actual_list = defaultdict(list)
    predicted_list = defaultdict(list)
    for fold in test_metrics:
        for pathway_name, metric in test_metrics[fold][ds_name][
            "pathway_metrics"
        ].items():
            actual = metric["actual"]
            predicted = metric["predicted"]
            actual_list[pathway_name].append(actual)
            predicted_list[pathway_name].append(predicted)

    pathway_actual_arrays = {}
    pathway_predicted_arrays = {}
    for pathway_name in actual_list.keys():
        actual = np.concatenate(actual_list[pathway_name])
        predicted = np.concatenate(predicted_list[pathway_name])
        # print(pathway_name, actual.shape, predicted.shape)
        pathway_actual_arrays[pathway_name] = actual
        pathway_predicted_arrays[pathway_name] = predicted
    return pathway_actual_arrays, pathway_predicted_arrays


def plot_scatter_for_all_pathways(test_metrics, ds_name, save_dir):
    pathway_actual_arrays, pathway_predicted_arrays = get_all_pathway_actual_predicted(
        test_metrics, ds_name
    )
    plot_pathway_predictions(
        pathway_actual_arrays,
        pathway_predicted_arrays,
        ds_name,
        save_dir,
    )
