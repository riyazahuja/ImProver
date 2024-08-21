# extract csv into pandas dataframe, extract accuracy (raw and nonzero),
# mean (blanks/0, raw/nonzero), median (blanks/0, raw/nonzero), stdev (blanks/0, raw/nonzero)
# for each combination.
# Additionally make a method for finding whether accuracy (whatever type) and improvement (?)
# differences between cases are statistically significant.
# Fisher's Exact Test for accuracy, Paired T-Test for improvement, certify normal
# distribution via Q-Q plot, or if not normal, use Wilcoxon Signed-Rank Test
# Also make a method to visualize (With error bars)

import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

import json


# Function to filter the dataframe based on specified parameters
def filter_data(df, **kwargs):
    valid = [
        "method",
        "n",
        "metric",
        "model",
        "annotation",
        "syntax_search",
        "mathlib_search",
        "examples",
    ]
    comp = True
    for k, v in kwargs.items():
        if k in valid:
            comp = comp & (df[k] == v)
    return df[comp]


# Function to calculate accuracy and improvement metrics
def calculate_metrics(filtered_df: pd.DataFrame, minimax="MAX"):
    accuracy = filtered_df["new_correct"].mean()

    time_mean = filtered_df["time"].mean()
    time_median = filtered_df["time"].median()
    time_stdev = filtered_df["time"].std()

    nonempty_deltas = filtered_df.dropna(subset=["delta"])
    zero_deltas = filtered_df.fillna(value={"delta": 0})

    if minimax == "MAX":
        improved_df = filtered_df[(filtered_df["delta"] > 0)]
        nonempty_improved_deltas = nonempty_deltas[(nonempty_deltas["delta"] > 0)][
            "delta"
        ]
        zero_improved_deltas = zero_deltas[(zero_deltas["delta"] > 0)]["delta"]
    else:
        improved_df = filtered_df[(filtered_df["delta"] < 0)]
        nonempty_improved_deltas = nonempty_deltas[(nonempty_deltas["delta"] < 0)][
            "delta"
        ]
        zero_improved_deltas = zero_deltas[(zero_deltas["delta"] < 0)]["delta"]

    nonzero_accuracy = len(improved_df["new_correct"]) / len(filtered_df)

    nonempty_deltas = nonempty_deltas["delta"]
    zero_deltas = zero_deltas["delta"]

    mean_improvement = nonempty_deltas.mean() if not nonempty_deltas.empty else None
    median_improvement = nonempty_deltas.median() if not nonempty_deltas.empty else None
    stdev_improvement = nonempty_deltas.std() if not nonempty_deltas.empty else None

    mean_zero_improvement = zero_deltas.mean() if not zero_deltas.empty else None
    median_zero_improvement = zero_deltas.median() if not zero_deltas.empty else None
    stdev_zero_improvement = zero_deltas.std() if not zero_deltas.empty else None

    mean_nonzero_improvement = (
        nonempty_improved_deltas.mean() if not nonempty_improved_deltas.empty else None
    )
    median_nonzero_improvement = (
        nonempty_improved_deltas.median()
        if not nonempty_improved_deltas.empty
        else None
    )
    stdev_nonzero_improvement = (
        nonempty_improved_deltas.std() if not nonempty_improved_deltas.empty else None
    )

    mean_zero_nonzero_improvement = (
        zero_improved_deltas.mean() if not zero_improved_deltas.empty else None
    )
    median_zero_nonzero_improvement = (
        zero_improved_deltas.median() if not zero_improved_deltas.empty else None
    )
    stdev_zero_nonzero_improvement = (
        zero_improved_deltas.std() if not zero_improved_deltas.empty else None
    )

    return {
        "accuracy": {"raw": accuracy, "nonzero": nonzero_accuracy},
        "mean_improvement": {
            ("nonempty", "raw"): mean_improvement,
            ("zero", "raw"): mean_zero_improvement,
            ("nonempty", "nonzero"): mean_nonzero_improvement,
            ("zero", "nonzero"): mean_zero_nonzero_improvement,
        },
        "median_improvement": {
            ("nonempty", "raw"): median_improvement,
            ("zero", "raw"): median_zero_improvement,
            ("nonempty", "nonzero"): median_nonzero_improvement,
            ("zero", "nonzero"): median_zero_nonzero_improvement,
        },
        "stdev_improvement": {
            ("nonempty", "raw"): stdev_improvement,
            ("zero", "raw"): stdev_zero_improvement,
            ("nonempty", "nonzero"): stdev_nonzero_improvement,
            ("zero", "nonzero"): stdev_zero_nonzero_improvement,
        },
        "time": {"mean": time_mean, "median": time_median, "stdev": time_stdev},
    }


def compare_nonzero_accuracy_pair(df, method1, method2):
    method1_data = filter_data(df, **method1)
    method2_data = filter_data(df, **method2)

    method1_data_filtered = method1_data[
        (method1_data["new_correct"] == True) & (method1_data["delta"] < 0)
    ]
    method2_data_filtered = method2_data[
        (method2_data["new_correct"] == True) & (method2_data["delta"] < 0)
    ]

    method1_success = method1_data_filtered["new_correct"].sum()
    method2_success = method2_data_filtered["new_correct"].sum()

    n1 = len(method1_data)
    n2 = len(method2_data)

    contingency_table = [
        [method1_success, n1 - method1_success],
        [method2_success, n2 - method2_success],
    ]
    _, p_value = stats.fisher_exact(contingency_table)

    return p_value


# Function to compare accuracy between two methods
def compare_accuracy_pair(df, method1, method2):
    method1_data = filter_data(df, **method1)
    method2_data = filter_data(df, **method2)

    method1_success = method1_data["new_correct"].sum()
    method2_success = method2_data["new_correct"].sum()

    n1 = len(method1_data)
    n2 = len(method2_data)

    contingency_table = [
        [method1_success, n1 - method1_success],
        [method2_success, n2 - method2_success],
    ]
    _, p_value = stats.fisher_exact(contingency_table)

    return p_value


# Function to check if the data is normally distributed using the Shapiro-Wilk test.
# Returns True if the data is normally distributed (p > 0.05), otherwise False.
def check_normality(data):

    if len(data) < 3:  # Shapiro-Wilk requires at least 3 data points
        return False
    _, p_value = stats.shapiro(data)
    return p_value > 0.05


# Function to compare improvement between two methods with normality check
def compare_improvement_pair(df, method1, method2):
    method1_data = filter_data(df, **method1)
    method2_data = filter_data(df, **method2)

    method1_deltas = method1_data["delta"].dropna()
    method2_deltas = method2_data["delta"].dropna()

    # Only proceed if both methods have enough data
    if len(method1_deltas) > 0 and len(method2_deltas) > 0:
        # Check normality
        normality_method1 = check_normality(method1_deltas)
        normality_method2 = check_normality(method2_deltas)

        if normality_method1 and normality_method2:
            # Use independent samples t-test if both datasets are normally distributed
            _, p_value = stats.ttest_ind(method1_deltas, method2_deltas)
        else:
            # Use Mann-Whitney U test if data is not normally distributed
            _, p_value = stats.mannwhitneyu(method1_deltas, method2_deltas)
    else:
        p_value = None

    return p_value


def compare_multiple_helper(df, fn, *methods):
    if len(methods) < 2:
        raise ValueError("not enough data provided for comparison")
    elif len(methods) == 2:
        return {(0, 1): compare_nonzero_accuracy_pair(df, methods[0], methods[1])}
    else:
        pairs = list(combinations(range(len(methods)), 2))
        p_values_raw = {
            pair: fn(df, methods[pair[0]], methods[pair[1]]) for pair in pairs
        }
        p_items = list(p_values_raw.items())

        p_items.sort(key=lambda x: x[1])
        trimmed = [item[1] for item in p_items]
        p_values_corrected_raw = multipletests(trimmed, is_sorted=True)[1]
        # print(len(trimmed))
        # print(len(p_values_corrected_raw))
        # print(p_values_corrected_raw)
        corrected_p_values = {
            p_items[i][0]: p_values_corrected_raw[i] for i in range(len(trimmed))
        }
        return corrected_p_values


def compare_nonzero_accuracy(df, *methods):
    return compare_multiple_helper(df, compare_nonzero_accuracy_pair, *methods)


def compare_accuracy(df, *methods):
    return compare_multiple_helper(df, compare_accuracy_pair, *methods)


def compare_improvement(df, *methods):
    return compare_multiple_helper(df, compare_improvement_pair, *methods)


def get_best_method_helper(
    df,
    methods,
    fn,
    get_metric_fn,
    alpha=0.05,
    minimax="MAX",
):

    metric_data = {
        i: get_metric_fn(calculate_metrics(filter_data(df, **methods[i])))
        for i in range(len(methods))
    }

    p_values = compare_multiple_helper(df, fn, *methods)
    significant = {pair: p_values[pair] < alpha for pair in p_values.keys()}

    best_raw = max(
        list(metric_data.items()), key=lambda x: x[1] if minimax == "MAX" else -1 * x[1]
    )

    best_method = best_raw[0]
    best_methods_significant = [
        methods[method]
        for method in range(len(methods))
        if (method != best_method)
        and (not significant.get((best_method, method), False))
        and (not significant.get((method, best_method), False))
    ] + [
        methods[best_method]
    ]  # all methods w/o sig diff from best

    if len(best_methods_significant) == 0:
        raise ValueError("no best method found")
    else:
        return best_methods_significant


def get_best_method_stats(
    df,
    methods,
    alpha=0.05,
    improvement_type=("mean_improvement", "nonempty", "raw"),
    time_type="mean",
    minimax="MAX",
):
    # strategy: get method with highest nonzero accuracy and find
    # all other methods that fail to reject null hypo
    # Then on this set of methods, get method with highest improvement (according to minimax)
    # and find all other methods that fail to reject null hypo
    # Then on this set of methods, get method with highest accuracy
    # and find all other methods that fail to reject null hypo
    # Then on this set of methods, get method that took the least amount of time.
    best_nonzero_accuracy_significant = get_best_method_helper(
        df,
        methods,
        fn=compare_nonzero_accuracy_pair,
        get_metric_fn=lambda x: x["accuracy"]["nonzero"],
        alpha=alpha,
    )
    if len(best_nonzero_accuracy_significant) == 1:
        return best_nonzero_accuracy_significant[0]
    print(
        f"Multiple best nonzero_accuracy:\n {best_nonzero_accuracy_significant}\n----------------"
    )

    best_improvement_significant = get_best_method_helper(
        df,
        best_nonzero_accuracy_significant,
        fn=compare_improvement_pair,
        get_metric_fn=lambda x: x[improvement_type[0]][
            (improvement_type[1], improvement_type[2])
        ],
        alpha=alpha,
        minimax=minimax,
    )

    if len(best_improvement_significant) == 1:
        return best_improvement_significant[0]

    print(
        f"Multiple best improvement:\n {best_improvement_significant}\n----------------"
    )

    best_accuracy_significant = get_best_method_helper(
        df,
        best_improvement_significant,
        fn=compare_accuracy_pair,
        get_metric_fn=lambda x: x["accuracy"]["raw"],
        alpha=alpha,
    )
    if len(best_accuracy_significant) == 1:
        return best_accuracy_significant[0]

    print(f"Multiple best accuracy:\n {best_accuracy_significant}\n----------------")

    time_data = {
        i: calculate_metrics(filter_data(df, **methods[i]))["time"][time_type]
        for i in range(len(methods))
    }

    best_time_raw = min(list(time_data.items()), key=lambda x: x[1])

    best_time_method = methods[best_time_raw[0]]
    return best_time_method


def get_best_method(df, methods, minimax="MAX"):
    get_metric_fn = lambda x: x["mean_improvement"][("zero", "raw")]
    metric_data = {
        i: get_metric_fn(calculate_metrics(filter_data(df, **methods[i])))
        for i in range(len(methods))
    }

    best_raw = max(
        list(metric_data.items()), key=lambda x: x[1] if minimax == "MAX" else -1 * x[1]
    )
    best_methods_i = [i for i, val in metric_data.items() if val == best_raw[1]]
    best_methods = [methods[i] for i in best_methods_i]
    if len(best_methods) == 1:
        return best_methods[0]
    print(f"Multiple best improvement:\n {best_methods}\n----------------")

    time_data = {
        i: calculate_metrics(filter_data(df, **methods[i]))["time"]["mean"]
        for i in range(len(methods))
    }

    best_time_raw = min(list(time_data.items()), key=lambda x: x[1])

    best_time_method = methods[best_time_raw[0]]
    return best_time_method


# Example function to plot nonzero accuracy, accuracy, and mean improvement with error bars
def plot_parameter_combinations(data_dict):
    # Extract relevant data for plotting
    parameter_combinations = list(data_dict.keys())
    nonzero_accuracies = [
        data_dict[param]["accuracy"]["nonzero"] for param in parameter_combinations
    ]
    accuracies = [
        data_dict[param]["accuracy"]["raw"] for param in parameter_combinations
    ]
    mean_improvements_nonempty_raw = [
        data_dict[param]["mean_improvement"][("nonempty", "raw")]
        for param in parameter_combinations
    ]
    mean_improvements_zero_raw = [
        data_dict[param]["mean_improvement"][("zero", "raw")]
        for param in parameter_combinations
    ]
    stdev_improvements_nonempty_raw = [
        data_dict[param]["stdev_improvement"][("nonempty", "raw")]
        for param in parameter_combinations
    ]
    stdev_improvements_zero_raw = [
        data_dict[param]["stdev_improvement"][("zero", "raw")]
        for param in parameter_combinations
    ]

    # Convert the parameter combinations to a format suitable for labeling (e.g., strings)
    labels = [str(param) for param in parameter_combinations]

    # Plot accuracy and nonzero accuracy
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(
        labels, nonzero_accuracies, alpha=0.7, label="Nonzero Accuracy", color="blue"
    )
    ax1.bar(labels, accuracies, alpha=0.5, label="Accuracy", color="orange")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy and Nonzero Accuracy Across Parameter Combinations")
    ax1.legend(loc="upper left")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Plot mean improvement with error bars for nonempty raw and zero raw
    fig, ax2 = plt.subplots(figsize=(10, 6))
    ax2.errorbar(
        labels,
        mean_improvements_nonempty_raw,
        yerr=stdev_improvements_nonempty_raw,
        fmt="o-",
        label="Mean Improvement (nonempty, raw)",
        color="green",
    )
    ax2.errorbar(
        labels,
        mean_improvements_zero_raw,
        yerr=stdev_improvements_zero_raw,
        fmt="s-",
        label="Mean Improvement (zero, raw)",
        color="red",
    )
    ax2.set_ylabel("Mean Improvement")
    ax2.set_title(
        "Mean Improvement with Standard Deviation Across Parameter Combinations"
    )
    ax2.legend(loc="upper left")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = "benchmark/data/parameter_tuning/final/Basic.csv"

    basic_methods = [
        {"method": fn, "annotation": anno}
        for fn in ["prompt_basic", "prompt_flat", "prompt_structured"]
        for anno in [True, False]
    ]
    # basic_methods = [{"examples": n} for n in [0, 3, 5, 7, 10]]
    # basic_methods = [
    #     {"method": fn}
    #     for fn in [
    #         "prompt_flat",
    #         "refinement(prompt_flat, prev_data_num=1, keep_best=False)",
    #         "best_of_n(prompt_flat)",
    #         "refinement(prompt_flat, prev_data_num=5, keep_best=False)",
    #         "refinement(prompt_flat, prev_data_num=1, keep_best=True)",
    #         "refinement(prompt_flat, prev_data_num=5, keep_best=True)",
    #     ]
    # ]
    # basic_methods = [
    #     {"model": model, "n": n}
    #     for model in ["gpt-4o", "gpt-4o-mini"]
    #     for n in [3, 5, 7, 10, 15]
    # ] + [{"model": "gpt-4o-mini", "n": 20}]
    # basic_methods = [
    #     {"method": method, "n": n, "mathlib_search": rag}
    #     for method in [
    #         "best_of_n(refinement_n(prompt_flat, prev_data_num=1, keep_best=True))",
    #         "refinement(best_of_n_n(prompt_flat), prev_data_num=1, keep_best=False)",
    #     ]
    #     for n in [3, 5]
    #     for rag in [True, False]
    # ] + [
    #     {"method": "best_of_n(prompt_flat)", "n": 15, "mathlib_search": rag}
    #     for rag in [True, False]
    # ]
    basic_df = pd.read_csv(file_path)

    data = [
        calculate_metrics(filter_data(basic_df, **method), minimax="MIN")
        for method in basic_methods
    ]
    for i in range(len(data)):
        print(basic_methods[i])
        print("****")
        print(data[i])
        print("====================")

    best_method = get_best_method(
        basic_df,
        basic_methods,
        minimax="MIN",
    )
    print(f"BEST:\n{best_method}")

    data_dict = {str(basic_methods[i]): data[i] for i in range(len(basic_methods))}
    plot_parameter_combinations(data_dict)
    # print(f'ORDER:\n{sort_methods(basic_df,basic_methods,minimax="MIN")}')
