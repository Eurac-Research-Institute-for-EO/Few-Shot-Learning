from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scaleagdata_vito.presto.presto_utils import get_feature_list, normalize_target


def prepare_data_for_cb(
    df: pd.DataFrame,
    target_name: str,
    lower_bound: Union[int, float, None] = None,
    upper_bound: Union[int, float, None] = None,
    num_time_steps: int = 36,
):
    feature_list = get_feature_list(num_time_steps)
    df_cx = df.loc[:, feature_list]
    if upper_bound is not None and lower_bound is not None:
        df_cy = normalize_target(
            df[target_name],
            upper_bound=upper_bound,
            lower_bound=lower_bound,
        )
    else:
        df_cy = df[target_name]
    # replace no_data value with nans for catboost to handle them differntly
    df_cx.replace(65535, None, inplace=True)
    return df_cx, df_cy


def prepare_cropland_data_for_presto(wc_df, sample_frac=0.005):
    def filter_remove_noncrops(df: pd.DataFrame) -> pd.DataFrame:
        crop_labels = [10, 11, 12, 13]
        df = df.loc[df.LANDCOVER_LABEL.isin(crop_labels)]
        return df

    wc_dataset = filter_remove_noncrops(wc_df).sample(frac=sample_frac)

    # convert the target variable to binary
    wc_dataset["LANDCOVER_LABEL"] = (wc_dataset["LANDCOVER_LABEL"] == 11).astype(int)
    return wc_dataset


def compare_performance_regression(metrics_raw_cb, metrics_presto_cb, metrics_presto):
    del metrics_presto_cb["RMSE"]
    del metrics_raw_cb["RMSE"]
    del metrics_presto["RMSE"]

    metrics = list(metrics_presto_cb.keys())
    values_dict1 = list(metrics_raw_cb.values())
    values_dict2 = list(metrics_presto_cb.values())
    values_dict3 = list(metrics_presto.values())

    x = np.arange(len(metrics))
    bar_width = 0.25
    fig, ax = plt.subplots()
    bars1 = ax.bar(
        x - bar_width, values_dict1, bar_width, label="Catboost", color="skyblue"
    )
    bars2 = ax.bar(x, values_dict2, bar_width, label="Presto + Cb", color="salmon")
    bars3 = ax.bar(
        x + bar_width, values_dict3, bar_width, label="Presto", color="lightgreen"
    )
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            yval = round(bar.get_height(), 2)
            ax.text(
                bar.get_x() + bar.get_width() / 2, yval, yval, ha="center", va="bottom"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Performance Comparison on Regression task")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    plt.show()


def compare_performance_classification(
    metrics_raw_cb, metrics_presto_cb, metrics_presto
):
    metrics = ["precision", "recall", "f1-score"]
    values_report1 = [metrics_raw_cb["macro avg"][metric] for metric in metrics]
    values_report2 = [metrics_presto_cb["macro avg"][metric] for metric in metrics]
    values_report3 = [metrics_presto["macro avg"][metric] for metric in metrics]

    x = np.arange(len(metrics))
    bar_width = 0.25
    fig, ax = plt.subplots()
    bars1 = ax.bar(
        x - bar_width, values_report1, bar_width, label="Catboost", color="skyblue"
    )
    bars2 = ax.bar(x, values_report2, bar_width, label="Presto + Cb", color="salmon")
    bars3 = ax.bar(
        x + bar_width, values_report3, bar_width, label="Presto", color="lightgreen"
    )
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            yval = round(bar.get_height(), 2)
            ax.text(
                bar.get_x() + bar.get_width() / 2, yval, yval, ha="center", va="bottom"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)

    ax.set_title("Performance Comparison on Classification task")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    plt.show()
