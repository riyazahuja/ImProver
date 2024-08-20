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

# Load the CSV file
file_path = "benchmark/data/final/basic.csv"
df = pd.read_csv(file_path)
print(df)
print(df.dtypes)

comp = (df["method"] == "prompt_flat") & (df["annotation"] == True)
new = df[comp]
print(new)


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
def calculate_metrics(filtered_df, minimax):
    accuracy = filtered_df["new_correct"].mean()

    time_mean = filtered_df["time"].mean()
    time_median = filtered_df["time"].median()
    time_stdev = filtered_df["time"].std()

    nonempty_deltas = filtered_df["delta"].dropna()
    zero_deltas = filtered_df["delta"].fillna(0)

    if minimax == "MAX":
        improved_df = filtered_df[(filtered_df["delta"] > 0)]
        nonempty_improved_deltas = nonempty_deltas[(nonempty_deltas["delta"] > 0)]
        zero_improved_deltas = zero_deltas[(zero_deltas["delta"] > 0)]
    else:
        improved_df = filtered_df[(filtered_df["delta"] < 0)]
        nonempty_improved_deltas = nonempty_deltas[(nonempty_deltas["delta"] < 0)]
        zero_improved_deltas = zero_deltas[(zero_deltas["delta"] < 0)]

    nonzero_accuracy = improved_df["new_correct"].mean()

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
        method1_data[("new_correct" == True) & ("delta" < 0)]
    ]
    method2_data_filtered = method2_data[
        method2_data[("new_correct" == True) & ("delta" < 0)]
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

    return {(method1, method2): p_value}


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

    return {(method1, method2): p_value}


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

    return {(method1, method2): p_value}


def compare_multiple_helper(df, *methods, fn):
    if len(methods) < 2:
        raise ValueError("not enough data provided for comparison")
    elif len(methods) == 2:
        return compare_nonzero_accuracy_pair(df, methods[0], methods[1])
    else:
        pairs = list(combinations(methods, 2))
        p_values_raw = {pair: fn(df, pair[0], pair[1]) for pair in pairs}
        coer_and_sort = [(k, p_values_raw[k]) for k in p_values_raw.keys()].sort(
            key=lambda x: x[1]
        )
        trimmed = [item[1] for item in coer_and_sort]
        p_values_corrected_raw = multipletests(trimmed, is_sorted=True)
        corrected_p_values = {
            coer_and_sort[i][0]: p_values_corrected_raw[i] for i in range(len(trimmed))
        }
        return corrected_p_values


def compare_nonzero_accuracy(df, *methods):
    return compare_multiple_helper(df, *methods, compare_nonzero_accuracy_pair)


def compare_accuracy(df, *methods):
    return compare_multiple_helper(df, *methods, compare_accuracy_pair)


def compare_improvement(df, *methods):
    return compare_multiple_helper(df, *methods, compare_improvement_pair)


def get_best_method_helper(
    df,
    methods,
    fn,
    get_metric_fn,
    alpha=0.05,
    minimax="MAX",
):

    metrics = {
        method: calculate_metrics(filter_data(df, **method)) for method in methods
    }

    metric_data = {method: get_metric_fn(metrics[method]) for method in metrics.keys()}
    p_values = compare_multiple_helper(df, *methods, fn)
    significant = {set(pair): p_values[pair] < alpha for pair in p_values.keys()}

    best_raw = max(
        list(metric_data.items()), key=lambda x: x[1] if minimax == "MAX" else -1 * x[1]
    )

    best_method = best_raw[0]
    best_methods_significant = [
        method
        for method in methods
        if (method != best_method) and (not significant[set((best_method, method))])
    ] + [
        best_method
    ]  # all methods w/o sig diff from best

    if len(best_methods_significant) == 0:
        raise ValueError("no best method found")
    else:
        return best_methods_significant


def get_best_method(
    df,
    methods,
    alpha=0.05,
    improvement_type=("mean_improvement", "nonzero", "raw"),
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

    best_accuracy_significant = get_best_method_helper(
        df,
        best_improvement_significant,
        fn=compare_accuracy_pair,
        get_metric_fn=lambda x: x["accuracy"]["raw"],
        alpha=alpha,
    )
    if len(best_accuracy_significant) == 1:
        return best_accuracy_significant[0]

    time_data = {
        method: calculate_metrics(filter_data(df, **method))["time"][time_type]
        for method in methods
    }

    best_time_raw = min(list(time_data.items()), key=lambda x: x[1])

    best_time_method = best_time_raw[0]
    return best_time_method


# Example usage
methods_to_compare = [
    "prompt_flat",
    "method_2",
    "method_3",
]  # Replace with actual method names
group_columns = [
    "method",
    "n",
    "metric",
    "model",
    "annotation",
    "syntax_search",
    "mathlib_search",
    "examples",
]


# Example usage
method1_params = {"method": "prompt_flat", "annotation": True}

# Filter data for each method
method1_data = filter_data(df, **method1_params)
print(method1_data)
# Calculate metrics
method1_metrics = calculate_metrics(method1_data)
print(method1_metrics)
