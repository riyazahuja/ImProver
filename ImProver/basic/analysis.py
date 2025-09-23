import re
from collections import defaultdict
import numpy as np
import argparse
import duckdb
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math


def get_db_connection(db_path):
    """Establishes a connection to a DuckDB database."""
    try:
        return duckdb.connect(
            database=str(db_path), read_only=False
        )  # Read_only False to allow creating tables if raw.duckdb
    except Exception as e:
        print(f"Error connecting to database {db_path}: {e}")
        return None


def load_config(config_path):
    """Loads the JSON configuration file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        if "metric" not in config or "n" not in config:
            print(f"Error: 'metric' or 'n' not found in config file {config_path}")
            return None
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}")
        return None
    except Exception as e:
        print(f"Error loading config {config_path}: {e}")
        return None


def parse_json_field_as_float(value):
    """Safely parses a JSON field (which might be a string or number) as float."""
    if value is None:
        return None
    try:
        f = float(value)
        # Handle inf and nan cases
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        # Try to parse as JSON, e.g., if value is a stringified number or single-element list
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list) and parsed:
                parsed = parsed[0]
            f = float(parsed)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except Exception:
            pass
        return None  # Or handle as an error, or return 0.0?


def get_delta(row):

    d = parse_json_field_as_float(row.get("delta"))
    if d is not None:
        return d
    return None

    # new_score = parse_json_field_as_float(row.get("new_score"))
    # old_score = parse_json_field_as_float(row.get("og_score"))
    # if new_score is None or old_score is None or old_score == 0:
    #     return None
    # return (new_score - old_score) / old_score


# --- Experimental Measures ---


def calculate_accuracy(rows):
    """Calculates accuracy: number of rows with new_correct as true / total rows."""
    if not rows:
        return 0.0
    correct_count = sum(1 for row in rows if row.get("new_correct"))
    return correct_count / len(rows)


def calculate_nonzero_accuracy(rows, metric_name, metric_spec_map):
    """
    Calculates nonzero accuracy:
    Number of rows with new_correct as true AND (delta < 0 if min, else delta > 0) / total rows.
    """
    if not rows:
        return 0.0

    metric_objective = metric_spec_map  # metric_spec_map.get(metric_name)
    if not metric_objective:
        print(
            f"Warning: Metric '{metric_name}' not found in METRIC_SPECIFICATIONS for nonzero_accuracy."
        )
        return 0.0  # Or handle error

    relevant_rows = 0
    for row in rows:
        delta = get_delta(row)
        # print(f"Row delta: {delta} (type: {type(delta)})")
        if row.get("new_correct") and delta is not None:
            if metric_objective == "min" and delta < 0:
                relevant_rows += 1
            elif metric_objective == "max" and delta > 0:
                relevant_rows += 1
    return relevant_rows / len(rows)


def calculate_improvement(rows, metric_name, metric_spec_map):
    """
    Calculates improvement: average delta.
    If delta is null, set to zero.
    If delta > 0 and metric is min, delta for that row is zero (and vice versa for max).
    """
    if not rows:
        return 0.0

    metric_objective = metric_spec_map  # metric_spec_map.get(metric_name)
    if not metric_objective:
        print(
            f"Warning: Metric '{metric_name}' not found in METRIC_SPECIFICATIONS for improvement."
        )
        return 0.0

    effective_deltas = []
    for row in rows:
        delta = get_delta(row)
        if delta is None:
            delta = 0.0

        if metric_objective == "min":
            effective_delta = delta if delta < 0 else 0.0
        elif metric_objective == "max":
            effective_delta = delta if delta > 0 else 0.0
        else:  # Should not happen if metric_objective is validated
            effective_delta = 0.0
        effective_deltas.append(effective_delta)

    return sum(effective_deltas) / len(rows) if effective_deltas else 0.0


def calculate_nonzero_improvement(rows, metric_name, metric_spec_map):
    """
    Calculates nonzero improvement:
    Average delta value across rows with delta not null AND (delta < 0 if min, else delta > 0).
    """
    metric_objective = metric_spec_map  # .get(metric_name)
    if not metric_objective:
        print(
            f"Warning: Metric '{metric_name}' not found in METRIC_SPECIFICATIONS for nonzero_improvement."
        )
        return 0.0

    relevant_deltas = []
    for row in rows:
        delta = get_delta(row)
        if delta is not None:
            if metric_objective == "min" and delta < 0:
                relevant_deltas.append(delta)
            elif metric_objective == "max" and delta > 0:
                relevant_deltas.append(delta)

    return sum(relevant_deltas) / len(relevant_deltas) if relevant_deltas else 0.0


# --- Analysis Commands ---
# make bon database, metric vs n graph, distribution shift graph, and pass/improvement rate graph.
# mark bon database entries with an improvement rate
def run_best_of_n_analysis(run_id, db_con, config, metric_objective):
    print("Starting Best-of-N analysis...")
    metric_name = config["metric"]
    n_config_val = int(config["n"])

    analysis_base_path = os.path.join("evals", run_id, "analysis", "BoN")
    os.makedirs(analysis_base_path, exist_ok=True)

    raw_db_path = os.path.join(analysis_base_path, "raw.duckdb")
    plot_path = os.path.join(analysis_base_path, "BoN.png")
    csv_path = os.path.join(analysis_base_path, "data.csv")
    hist_og_vs_new_path = os.path.join(analysis_base_path, "score_distribution.png")
    hist_imprate_path = os.path.join(
        analysis_base_path, "improvement_rate_distribution.png"
    )

    tick_size = max(math.floor(n_config_val / 16), 1)
    tick_n_values = [1]
    for i in range(1, 16):
        val = i * tick_size
        if val < n_config_val and val not in tick_n_values:
            tick_n_values.append(val)
    if n_config_val not in tick_n_values:
        tick_n_values.append(n_config_val)
    tick_n_values = sorted(list(set(tick_n_values)))

    print(
        f"Config: metric='{metric_name}' (objective: {metric_objective}), N={n_config_val}, tick_size={tick_size}"
    )
    print(f"Tick n values for analysis: {tick_n_values}")

    try:
        distinct_pairs_query = "SELECT DISTINCT decl, module FROM evaluation_results"
        distinct_pairs = db_con.execute(distinct_pairs_query).fetchall()
        if not distinct_pairs:
            print("No (decl, module) pairs found in eval.duckdb. Aborting BoN.")
            return False

        all_data_query = (
            "SELECT original_prompt,og_score,og_raw,"
            "list_transform(og_errors, x -> CAST(x as VARCHAR))::VARCHAR[] AS og_errors,"
            "new_trimmed, new_score,new_raw,"
            "list_transform(new_errors, x -> CAST(x as VARCHAR))::VARCHAR[] AS new_errors,"
            "new_correct, module, delta, decl,rowid "
            "FROM evaluation_results ORDER BY module, decl, rowid"
        )
        all_data_df = db_con.execute(all_data_query).fetchdf()
        all_data_rows = all_data_df.to_dict("records")
    except Exception as e:
        print(f"Database error during BoN setup: {e}")
        return False

    # Calculate full_accuracy and full_improvement across all rows
    total_correct_count = sum(1 for row in all_data_rows if row.get("new_correct"))
    full_accuracy = total_correct_count / len(all_data_rows) if all_data_rows else 0.0

    total_delta_sum = 0.0
    for row in all_data_rows:
        delta = get_delta(row)
        if metric_objective == "min" and delta is not None and delta < 0:
            total_delta_sum += delta
        elif metric_objective == "max" and delta is not None and delta > 0:
            total_delta_sum += delta
        # null values are counted as zero (no need to add 0)
    full_improvement = total_delta_sum / len(all_data_rows) if all_data_rows else 0.0

    graph_data_points = []
    collected_rows_for_raw_db = []
    collected_improvement_rates = []
    collected_og_scores = []
    collected_new_scores = []

    # Pre-group data by (decl, module)
    grouped_data = {}
    for row in all_data_rows:
        pair = (row["decl"], row["module"])
        if pair not in grouped_data:
            grouped_data[pair] = []
        grouped_data[pair].append(row)

    for n_val in tick_n_values:
        current_tick_best_rows_for_measures = []

        for decl_val, module_val in distinct_pairs:
            pair_key = (decl_val, module_val)
            L_dm_all_for_pair = grouped_data.get(pair_key, [])
            sub_list_for_dm = L_dm_all_for_pair[:n_val]

            if not sub_list_for_dm:
                continue

            # Rows that are correct and have a parsable delta
            correct_rows = [r for r in sub_list_for_dm if r.get("new_correct")]

            correct_rows_with_delta = [
                r for r in correct_rows if get_delta(r) is not None
            ]

            best_row_in_sub_list = None
            if correct_rows_with_delta:
                if metric_objective == "min":
                    best_row_in_sub_list = min(
                        correct_rows_with_delta, key=lambda r: get_delta(r)
                    )
                else:  # "max"
                    best_row_in_sub_list = max(
                        correct_rows_with_delta, key=lambda r: get_delta(r)
                    )
            elif correct_rows:
                correct_rows_with_score = [
                    r
                    for r in correct_rows
                    if parse_json_field_as_float(r.get("new_score")) is not None
                ]
                if correct_rows_with_score:
                    best_row_in_sub_list = correct_rows_with_score[0]
                else:
                    best_row_in_sub_list = correct_rows[0]
            else:
                best_row_in_sub_list = sub_list_for_dm[0]

            if best_row_in_sub_list:
                current_tick_best_rows_for_measures.append(best_row_in_sub_list)
                if n_val == n_config_val:
                    # Calculate improvement rate for this (decl, module) group
                    count_improved = 0
                    for r in sub_list_for_dm:
                        if r.get("new_correct"):
                            delta = get_delta(r)
                            if delta is not None:
                                if metric_objective == "min" and delta < 0:
                                    count_improved += 1
                                elif metric_objective == "max" and delta > 0:
                                    count_improved += 1
                    improvement_rate = (
                        count_improved / n_config_val if n_config_val > 0 else 0.0
                    )
                    # Add improvement_rate to the row for raw.duckdb
                    row_for_raw = {
                        k: v for k, v in best_row_in_sub_list.items() if k != "rowid"
                    }
                    row_for_raw["improvement_rate"] = improvement_rate
                    collected_rows_for_raw_db.append(row_for_raw)
                    collected_improvement_rates.append(improvement_rate)
                    # For histogram: og_score and new_score
                    og_score = parse_json_field_as_float(
                        best_row_in_sub_list.get("og_score")
                    )
                    new_score = parse_json_field_as_float(
                        best_row_in_sub_list.get("new_score")
                    )
                    delta = get_delta(best_row_in_sub_list)
                    # For the distribution shift plot, only count new_score as "new" if it is an improvement (delta<0 for min, delta>0 for max), else use og_score
                    if og_score is not None:
                        collected_og_scores.append(og_score)
                        if new_score is not None and delta is not None:
                            if (metric_objective == "min" and delta < 0) or (
                                metric_objective == "max" and delta > 0
                            ):
                                collected_new_scores.append(new_score)
                            else:
                                collected_new_scores.append(og_score)
                        else:
                            # If new_score is None, treat as og_score for overlay
                            collected_new_scores.append(og_score)
                    # If og_score is None, skip both for this item

        # Calculate measures for the current_tick_best_rows_for_measures
        if current_tick_best_rows_for_measures:
            acc = calculate_accuracy(current_tick_best_rows_for_measures)
            non_zero_acc = calculate_nonzero_accuracy(
                current_tick_best_rows_for_measures, metric_name, metric_objective
            )
            impr = calculate_improvement(
                current_tick_best_rows_for_measures, metric_name, metric_objective
            )
            non_zero_impr = calculate_nonzero_improvement(
                current_tick_best_rows_for_measures, metric_name, metric_objective
            )

            graph_data_points.append(
                {
                    "n_value": n_val,
                    "accuracy": acc,
                    "nonzero_accuracy": non_zero_acc,
                    "improvement": impr,
                    "nonzero_improvement": non_zero_impr,
                    "full_accuracy": full_accuracy,
                    "full_improvement": full_improvement,
                }
            )
            print(
                f"  n_val={n_val}: Acc={acc:.3f}, NonZeroAcc={non_zero_acc:.3f}, Impr={impr:.3f}, NonZeroImpr={non_zero_impr:.3f} (from {len(current_tick_best_rows_for_measures)} best rows)"
            )
        else:
            print(f"  n_val={n_val}: No best rows found to calculate measures.")
            graph_data_points.append(
                {
                    "n_value": n_val,
                    "accuracy": 0,
                    "nonzero_accuracy": 0,
                    "improvement": 0,
                    "nonzero_improvement": 0,
                    "full_accuracy": full_accuracy,
                    "full_improvement": full_improvement,
                }
            )

    # Save collected_rows_for_raw_db to raw.duckdb
    if collected_rows_for_raw_db:
        print(f"Saving {len(collected_rows_for_raw_db)} rows to {raw_db_path}...")
        df_raw = pd.DataFrame(collected_rows_for_raw_db)
        try:
            if os.path.exists(raw_db_path):
                os.remove(raw_db_path)
            con_raw = duckdb.connect(database=str(raw_db_path))
            con_raw.execute("CREATE TABLE best_results AS SELECT * FROM df_raw")
            con_raw.close()
            print(f"Successfully saved to {raw_db_path}")
        except Exception as e:
            print(f"Error saving to raw.duckdb: {e}")
            return False
    else:
        print("No rows collected for raw.duckdb.")

    # Create and save CSV and plot
    if graph_data_points:
        df_graph = pd.DataFrame(graph_data_points)
        try:
            df_graph.to_csv(csv_path, index=False)
            print(f"Analysis data saved to {csv_path}")
            multiplier = -1 if metric_objective == "min" else 1
            plt.figure(figsize=(12, 7))
            plt.plot(
                df_graph["n_value"], df_graph["accuracy"], marker="o", label="Accuracy"
            )
            plt.plot(
                df_graph["n_value"],
                multiplier * df_graph["improvement"],
                marker="^",
                label="Improvement",
            )
            plt.xlabel("n Value (Number of samples considered per (decl,module))")
            plt.ylabel("Metric Value")
            plt.title(
                f"Best-of-N Analysis (Metric: {metric_name}, N_config: {n_config_val})"
            )
            plt.legend()
            plt.grid(True)
            plt.ylim(
                0,
                max(
                    1,
                    max(
                        df_graph["accuracy"].max(),
                        multiplier * df_graph["improvement"].max(),
                    ),
                )
                * 1.1,
            )
            plt.xticks(tick_n_values)
            plt.tight_layout()
            plt.savefig(plot_path)
            print(f"Analysis plot saved to {plot_path}")
            plt.close()
        except Exception as e:
            print(f"Error generating CSV/plot: {e}")
    else:
        print("No data points for graph and CSV.")

    # --- Additional Plots: Score Distribution and Improvement Rate Histogram ---

    # 1. Enhanced Histogram of og_score and new_score (distribution shift)
    if collected_og_scores and collected_new_scores:
        try:
            import numpy as np

            plt.figure(figsize=(14, 8))
            bins = 30

            # Compute means and medians
            og_mean = np.mean(collected_og_scores)
            new_mean = np.mean(collected_new_scores)
            og_median = np.median(collected_og_scores)
            new_median = np.median(collected_new_scores)

            # Plot side-by-side (grouped) histograms for original and new scores

            # Compute common bin edges for both histograms
            all_scores = collected_og_scores + collected_new_scores
            bins_edges = np.histogram_bin_edges(all_scores, bins=bins)

            # Compute histogram counts for each group
            og_hist, _ = np.histogram(
                collected_og_scores, bins=bins_edges, density=True
            )
            new_hist, _ = np.histogram(
                collected_new_scores, bins=bins_edges, density=True
            )

            # Compute bin centers and width for bar plotting
            bin_centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
            width = (
                bins_edges[1] - bins_edges[0]
            ) * 0.4  # 40% of bin width for each bar

            # Plot side-by-side bars
            plt.bar(
                bin_centers - width / 2,
                og_hist,
                width=width,
                alpha=0.8,
                label="Original Score",
                color="blue",
                edgecolor="black",
            )
            plt.bar(
                bin_centers + width / 2,
                new_hist,
                width=width,
                alpha=0.8,
                label="New Score",
                color="orange",
                edgecolor="black",
            )

            # Plot mean and median lines
            plt.axvline(
                og_mean,
                color="blue",
                linestyle="--",
                linewidth=2,
                label=f"Original Mean: {og_mean:.2f}",
            )
            plt.axvline(
                new_mean,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"New Mean: {new_mean:.2f}",
            )
            plt.axvline(
                og_median,
                color="blue",
                linestyle=":",
                linewidth=2,
                label=f"Original Median: {og_median:.2f}",
            )
            plt.axvline(
                new_median,
                color="orange",
                linestyle=":",
                linewidth=2,
                label=f"New Median: {new_median:.2f}",
            )

            # Annotate means and medians
            plt.text(
                og_mean,
                plt.ylim()[1] * 0.95,
                f"{og_mean:.2f}",
                color="blue",
                ha="right",
                va="top",
                fontsize=10,
                rotation=90,
            )
            plt.text(
                new_mean,
                plt.ylim()[1] * 0.95,
                f"{new_mean:.2f}",
                color="orange",
                ha="left",
                va="top",
                fontsize=10,
                rotation=90,
            )
            plt.text(
                og_median,
                plt.ylim()[1] * 0.85,
                f"{og_median:.2f}",
                color="blue",
                ha="right",
                va="top",
                fontsize=10,
                rotation=90,
            )
            plt.text(
                new_median,
                plt.ylim()[1] * 0.85,
                f"{new_median:.2f}",
                color="orange",
                ha="left",
                va="top",
                fontsize=10,
                rotation=90,
            )

            # Optionally, show arrows for mean shift
            plt.annotate(
                "",
                xy=(new_mean, plt.ylim()[1] * 0.8),
                xytext=(og_mean, plt.ylim()[1] * 0.8),
                arrowprops=dict(
                    facecolor="gray", shrink=0.05, width=2, headwidth=8, alpha=0.5
                ),
            )
            plt.text(
                (og_mean + new_mean) / 2,
                plt.ylim()[1] * 0.82,
                "Mean Shift",
                color="gray",
                ha="center",
                fontsize=10,
            )

            plt.xlabel("Score")
            plt.ylabel("Density")
            plt.title(
                f"Score Distribution Shift (Best-of-N, N={n_config_val})\n"
                f"Original Mean: {og_mean:.2f}, New Mean: {new_mean:.2f} | "
                f"Original Median: {og_median:.2f}, New Median: {new_median:.2f}"
            )
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(hist_og_vs_new_path)
            print(f"Score distribution plot saved to {hist_og_vs_new_path}")
            plt.close()
        except Exception as e:
            print(f"Error generating score distribution plot: {e}")
    else:
        print("Not enough data for score distribution plot.")

    # 2. Histogram and curve of improvement rates
    if collected_improvement_rates:
        try:
            import numpy as np

            plt.figure(figsize=(14, 8))
            bins = 20

            # Histogram
            n, bins_edges, patches = plt.hist(
                collected_improvement_rates,
                bins=bins,
                alpha=0.7,
                color="green",
                edgecolor="black",
                density=True,
                label="Improvement Rate Histogram",
            )

            # Mean and median
            imp_mean = np.mean(collected_improvement_rates)
            imp_median = np.median(collected_improvement_rates)

            # Plot mean and median lines
            plt.axvline(
                imp_mean,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {imp_mean:.2f}",
            )
            plt.axvline(
                imp_median,
                color="purple",
                linestyle=":",
                linewidth=2,
                label=f"Median: {imp_median:.2f}",
            )

            # Annotate mean and median
            plt.text(
                imp_mean,
                plt.ylim()[1] * 0.95,
                f"{imp_mean:.2f}",
                color="red",
                ha="right",
                va="top",
                fontsize=10,
                rotation=90,
            )
            plt.text(
                imp_median,
                plt.ylim()[1] * 0.85,
                f"{imp_median:.2f}",
                color="purple",
                ha="left",
                va="top",
                fontsize=10,
                rotation=90,
            )

            # Optionally, plot a smoothed curve (moving average) over the histogram
            # We'll use a simple moving average of the histogram values for a "curve"
            bin_centers = 0.5 * (bins_edges[1:] + bins_edges[:-1])
            if len(n) > 2:
                window = max(2, int(len(n) / 8))
                smooth = np.convolve(n, np.ones(window) / window, mode="same")
                plt.plot(
                    bin_centers,
                    smooth,
                    color="black",
                    linewidth=2,
                    label="Smoothed Curve",
                )

            plt.xlabel("Improvement Rate")
            plt.ylabel("Density")
            plt.title(
                f"Improvement Rate Distribution (Best-of-N, N={n_config_val})\n"
                f"Mean: {imp_mean:.2f}, Median: {imp_median:.2f}"
            )
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(hist_imprate_path)
            print(f"Improvement rate distribution plot saved to {hist_imprate_path}")
            plt.close()
        except Exception as e:
            print(f"Error generating improvement rate distribution plot: {e}")
    else:
        print("Not enough data for improvement rate distribution plot.")

    # 3. Average Score Distribution Relative to Best per Theorem
    if all_data_rows:
        try:

            # Group all original rows by (module, decl) to analyze per theorem
            theorem_groups = defaultdict(list)
            for row in all_data_rows:
                key = (row["module"], row["decl"])
                # Only take first n_config_val samples for each theorem
                if len(theorem_groups[key]) < n_config_val:
                    theorem_groups[key].append(row)

            # Calculate relative score distributions for each theorem
            all_relative_distributions = []
            null_counts = []

            for theorem_key, theorem_rows in theorem_groups.items():
                # Extract scores and find the best score for this theorem
                scores = []
                null_count = 0

                for row in theorem_rows:
                    score = parse_json_field_as_float(row.get("new_score"))
                    if score is not None:
                        scores.append(score)
                    else:
                        null_count += 1

                if not scores:
                    # If all scores are null for this theorem, skip
                    continue

                # Find best score based on metric objective
                if metric_objective == "min":
                    best_score = min(scores)
                else:  # "max"
                    best_score = max(scores)

                # Calculate relative percentages for this theorem
                relative_scores = []
                # if best_score != 0:

                for score in scores:

                    relative_pct = multiplier * (score - best_score)

                    relative_scores.append(relative_pct)
                # else:
                #     # Handle case where best_score is 0
                #     relative_scores = [100.0] * len(scores)

                all_relative_distributions.append(relative_scores)
                null_counts.append(null_count)

            if all_relative_distributions:
                # Flatten all relative scores to determine overall distribution
                all_relative_scores = []
                for rel_scores in all_relative_distributions:
                    all_relative_scores.extend(rel_scores)

                if all_relative_scores:
                    # Calculate dynamic bin edges based on the distribution
                    min_score = min(all_relative_scores)
                    max_score = max(all_relative_scores)

                    # Create bins with the best score (0) always being one of the bin edges
                    if min_score >= 0:
                        # All scores are non-negative (better than or equal to best)
                        bins = np.linspace(0, max_score * 1.1, 20)
                    else:
                        # Some scores are negative (worse than best)
                        # Ensure 0 is a bin edge and create symmetric or asymmetric bins
                        if max_score <= 0:
                            # All scores are non-positive
                            bins = np.linspace(min_score * 1.1, 0, 20)
                        else:
                            # Scores span both sides of 0
                            # Create more bins on the side with larger range
                            neg_range = abs(min_score)
                            pos_range = max_score

                            if neg_range > pos_range:
                                neg_bins = np.linspace(min_score * 1.1, 0, 15)
                                pos_bins = np.linspace(0, max_score * 1.1, 6)[
                                    1:
                                ]  # Exclude 0 to avoid duplication
                            else:
                                neg_bins = np.linspace(min_score * 1.1, 0, 6)
                                pos_bins = np.linspace(0, max_score * 1.1, 15)[
                                    1:
                                ]  # Exclude 0 to avoid duplication

                            bins = np.concatenate([neg_bins, pos_bins])

                    # Calculate average distribution across all theorems
                    theorem_histograms = []
                    total_samples_per_theorem = []

                    for i, rel_scores in enumerate(all_relative_distributions):
                        hist, _ = np.histogram(rel_scores, bins=bins, density=False)
                        total_samples = len(rel_scores) + null_counts[i]
                        if total_samples > 0:
                            # Normalize by total samples in this theorem (including nulls)
                            hist_normalized = hist / total_samples
                            theorem_histograms.append(hist_normalized)
                            total_samples_per_theorem.append(total_samples)

                    if theorem_histograms:
                        # Calculate average across all theorems
                        avg_hist = np.mean(theorem_histograms, axis=0)

                        # Calculate average null percentage
                        avg_null_pct = np.mean(
                            [
                                null_counts[i]
                                / (len(all_relative_distributions[i]) + null_counts[i])
                                for i in range(len(all_relative_distributions))
                                if (len(all_relative_distributions[i]) + null_counts[i])
                                > 0
                            ]
                        )

                        # Create the plot
                        plt.figure(figsize=(14, 8))

                        # Plot histogram bars
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        widths = bins[1:] - bins[:-1]

                        # Color bars based on whether they represent improvements (positive) or degradations (negative)
                        colors = [
                            (
                                "lightcoral"
                                if center < 0
                                else "lightgreen" if center > 0 else "gold"
                            )
                            for center in bin_centers
                        ]

                        plt.bar(
                            bin_centers,
                            avg_hist,
                            width=widths * 0.8,
                            alpha=0.7,
                            color=colors,
                            edgecolor="black",
                            label="Score Distribution",
                        )

                        # Add a vertical line at 0 (best score)
                        plt.axvline(
                            x=0,
                            color="black",
                            linestyle="--",
                            linewidth=2,
                            label="Best Score (0)",
                        )

                        # Add null bar separately on the right side
                        null_x = max(bins) + (max(bins) - min(bins)) * 0.1
                        null_width = (max(bins) - min(bins)) * 0.05
                        plt.bar(
                            null_x,
                            avg_null_pct,
                            width=null_width,
                            alpha=0.7,
                            color="gray",
                            edgecolor="black",
                            label="Null Scores",
                        )

                        # Add statistics text
                        total_theorems = len(theorem_histograms)
                        avg_samples_per_theorem = np.mean(total_samples_per_theorem)

                        plt.text(
                            0.02,
                            0.98,
                            f"Theorems analyzed: {total_theorems}\n"
                            f"Avg samples per theorem: {avg_samples_per_theorem:.1f}\n"
                            f"Avg null percentage: {avg_null_pct:.1%}\n"
                            f"Score range: [{min_score:.3f}, {max_score:.3f}]",
                            transform=plt.gca().transAxes,
                            fontsize=10,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                        )

                        plt.xlabel(
                            "Score Difference from Best (0 = Best, Negative = Worse)"
                        )
                        plt.ylabel("Average Fraction of Samples")
                        plt.title(
                            f"Average Score Distribution Relative to Best per Theorem\n"
                            f"(Metric: {metric_name}, Objective: {metric_objective}, N={n_config_val})"
                        )

                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()

                        # Save plot
                        relative_dist_path = os.path.join(
                            analysis_base_path, "relative_score_distribution.png"
                        )
                        plt.savefig(relative_dist_path)
                        print(
                            f"Relative score distribution plot saved to {relative_dist_path}"
                        )
                        plt.close()
                    else:
                        print("No valid theorem histograms to average.")
                else:
                    print("No relative scores found to create distribution.")
            else:
                print("No relative score distributions calculated.")

        except Exception as e:
            print(f"Error generating relative score distribution plot: {e}")
    else:
        print("No data available for relative score distribution plot.")

    print("Best-of-N analysis finished.")

    return True


def make_training_data_json(run_id, db_path, n, metric_objective):
    """
    Generates a training_data.json file as specified in the prompt.
    Args:
        db_path (str): Path to eval.duckdb database.
        n (int): Best-of-n value.
        metric_objective (str): 'min' or 'max'.
    """
    import json

    # Set delta_multiplier
    delta_multiplier = 1 if metric_objective == "max" else -1

    # Connect to DuckDB and fetch all rows
    try:
        con = duckdb.connect(database=str(db_path), read_only=True)
        query = (
            "SELECT original_prompt, og_raw, new_trimmed, new_raw, new_correct, module, decl, delta "
            "FROM evaluation_results "
            "ORDER BY module, decl, rowid"
        )
        df = con.execute(query).fetchdf()
        con.close()
    except Exception as e:
        print(f"Error reading from database {db_path}: {e}")
        return

    # Group rows by (decl, module)

    grouped = defaultdict(list)
    for row in df.to_dict("records"):
        key = (row["module"], row["decl"])
        grouped[key].append(row)

    # Only keep groups with exactly n rows
    filtered_grouped = {k: v for k, v in grouped.items() if len(v) == n}

    training_data = {}

    for key, rows in filtered_grouped.items():
        module, decl = key
        prompt = rows[0].get("original_prompt")
        original = rows[0].get("og_raw")

        # Calculate pass_rate and improvement_rate
        pass_count = sum(1 for r in rows if r.get("new_correct"))
        pass_rate = pass_count / n if n > 0 else 0.0

        improvement_count = 0
        for r in rows:
            delta = r.get("delta")
            if r.get("new_correct") and delta is not None:
                try:
                    delta_val = float(delta)
                except Exception:
                    continue
                if delta_multiplier * delta_val > 0:
                    improvement_count += 1
        improvement_rate = improvement_count / n if n > 0 else 0.0

        # Subset of rows with new_correct==True and delta_multiplier*delta > 0
        valid_rows = []
        for r in rows:
            if r.get("new_correct"):
                delta = r.get("delta")
                if delta is not None:
                    try:
                        delta_val = float(delta)
                    except Exception:
                        continue
                    if delta_multiplier * delta_val > 0:
                        valid_rows.append((delta_multiplier * delta_val, r))

        # Sort valid_rows by delta_multiplier*delta descending
        valid_rows_sorted = sorted(valid_rows, key=lambda x: x[0], reverse=True)

        # Champion
        # if len(valid_rows_sorted) >= 1:
        #     champion_row = valid_rows_sorted[0][1]
        #     champion = {
        #         "output": champion_row.get("new_trimmed"),
        #         "cot_output": champion_row.get("new_raw"),
        #         "delta": valid_rows_sorted[0][0]
        #     }
        # else:
        #     champion = {"output": None, "cot_output": None, "delta": None}

        output_valid = []
        for delta, row in valid_rows_sorted:
            output_valid.append(
                {
                    "output": row.get("new_trimmed"),
                    "cot_output": row.get("new_raw"),
                    "delta": delta,
                }
            )

        # Invalid: any row with new_correct==False
        output_invalid = []
        for row in [r for r in rows if not r.get("new_correct")]:
            output_invalid.append(
                {
                    "output": row.get("new_trimmed"),
                    "cot_output": row.get("new_raw"),
                    "delta": None,
                }
            )

        training_data[f"{module}_{decl}"] = {
            "prompt": prompt,
            "original": original,
            "improvement_rate": improvement_rate,
            "pass_rate": pass_rate,
            "valid_samples": output_valid,
            "invalid_samples": output_invalid,
        }

    # Save to training_data.json in the same directory as db_path
    out_dir = os.path.join("evals", run_id, "analysis", "BoN")
    out_path = os.path.join(out_dir, "training_data.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        print(f"Saved training data to {out_path}")
    except Exception as e:
        print(f"Error saving training_data.json: {e}")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Perform analysis on experimental run data."
    )
    parser.add_argument("run_id", help="Identifier for the run.")

    return parser


def main(args):

    db_path = os.path.join("evals", args.run_id, "eval.duckdb")
    config_path = os.path.join("evals", args.run_id, "config.json")

    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return

    config = load_config(config_path)
    if not config:
        return

    metric = config.get("metric")
    metric_config_path = os.path.join("metrics", metric, "config.json")
    try:
        with open(metric_config_path, "r") as f:
            metric_config = json.load(f)
        metric_objective = metric_config.get("scoring", {}).get("minmax", "min")
    except FileNotFoundError:
        print(f"Error: Metric config file not found at {metric_config_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {metric_config_path}")
        return
    except Exception as e:
        print(f"Error loading metric config {metric_config_path}: {e}")
        return

    # Main DB connection for eval.duckdb (used by BoN if called directly)
    # Training command might re-open it if BoN needs to be run first.
    # BoN itself also opens a connection to raw.duckdb.

    # For BoN, we pass the main eval.duckdb connection.
    # For Training, it handles its own connections (to raw.duckdb, and to eval.duckdb if BoN needs to run).

    db_con_eval = get_db_connection(db_path)

    # make bon database, metric vs n graph, distribution shift graph, and pass/improvement rate graph.
    # mark bon database entries with a pass rate and improvement rate
    run_best_of_n_analysis(args.run_id, db_con_eval, config, metric_objective)
    db_con_eval.close()

    make_training_data_json(args.run_id, db_path, int(config["n"]), metric_objective)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
