#!/usr/bin/env python3
"""
Analyze ablation study results and generate LaTeX tables and plots.

This script processes evaluation results from evals/ablations/ and:
1. Generates LaTeX tables showing n=8 metrics for each experiment group
2. Creates plots showing accuracy and improvement vs n (1-8) for each group

Usage:
    python analyze_ablations.py [--evals-dir PATH] [--output-dir PATH] [--tables-only] [--graphs-only]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


def load_run_data(run_path: Path) -> pd.DataFrame:
    """Load analysis/BoN/data.csv for a single run."""
    csv_path = run_path / "analysis" / "BoN" / "data.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Warning: Failed to load {csv_path}: {e}")
        return None


def scan_ablation_runs(evals_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Scan evals/ablations/ directory and load all run data.

    Returns:
        Dict mapping folder names to dict of {run_name: dataframe}
    """
    results = {}

    if not evals_dir.exists():
        print(f"Error: Directory {evals_dir} does not exist")
        return results

    for folder in sorted(evals_dir.iterdir()):
        if not folder.is_dir():
            continue

        folder_runs = {}
        for run_dir in sorted(folder.iterdir()):
            if not run_dir.is_dir():
                continue

            df = load_run_data(run_dir)
            if df is not None:
                folder_runs[run_dir.name] = df

        if folder_runs:
            results[folder.name] = folder_runs
            print(f"Loaded {len(folder_runs)} runs from {folder.name}")

    return results


def categorize_first_iter_sft_wsft(runs: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict]:
    """Categorize first iteration SFT/wSFT runs into lr sweep and vt sweep."""
    lr_sweep = {}
    vt_sweep = {}

    for run_name, df in runs.items():
        if run_name.startswith("SFT_lr") or run_name.startswith("wSFT_lr"):
            lr_sweep[run_name] = df
        elif run_name.startswith("wSFT_vt"):
            vt_sweep[run_name] = df

    return lr_sweep, vt_sweep


def categorize_first_iter_irpo_dpo(
    runs: Dict[str, pd.DataFrame],
) -> Tuple[Dict, Dict, Dict]:
    """Categorize first iteration IRPO/DPO runs into beta/alpha, lr, and w/l sweeps."""
    beta_alpha = {}
    lr_sweep = {}
    wl_sweep = {}

    for run_name, df in runs.items():
        if "gap" in run_name:
            # Skip gap variants
            continue
        elif run_name.startswith("IRPO_beta") or run_name.startswith("DPO_beta"):
            beta_alpha[run_name] = df
        elif run_name.startswith("IRPO_lr"):
            lr_sweep[run_name] = df
        elif run_name.startswith("IRPO_w") and "_l" in run_name:
            wl_sweep[run_name] = df

    return beta_alpha, lr_sweep, wl_sweep


def extract_n8_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Extract metrics for n=8 from a dataframe."""
    n8_row = df[df["n_value"] == 8]
    if len(n8_row) == 0:
        return None

    row = n8_row.iloc[0]
    return {
        "accuracy": row["accuracy"] * 100,  # Convert to percentage
        "nonzero_accuracy": row["nonzero_accuracy"] * 100,
        "improvement": row["improvement"],
        "nonzero_improvement": row["nonzero_improvement"],
    }


def format_latex_table(
    data: Dict[str, Dict[str, float]], caption: str, label: str
) -> str:
    """Generate LaTeX table from metrics data."""
    if not data:
        return ""

    # Create DataFrame
    rows = []
    for run_name, metrics in data.items():
        if metrics is None:
            continue
        row = {
            "Method": run_name,
            "Accuracy": f"{metrics['accuracy']:.2f}",
            "Nonzero Acc.": f"{metrics['nonzero_accuracy']:.2f}",
            "Improvement": f"{metrics['improvement']:.4f}",
            "Nonzero Imp.": f"{metrics['nonzero_improvement']:.4f}",
        }
        rows.append(row)

    if not rows:
        return ""

    df = pd.DataFrame(rows)

    # Find best values (bold them)
    # For accuracy: higher is better
    # For improvement: more negative is better (length minimization)

    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append(f"\\label{{{label}}}")
    latex_lines.append("\\begin{tabular}{l" + "r" * (len(df.columns) - 1) + "}")
    latex_lines.append("\\toprule")

    # Header
    header = " & ".join(df.columns) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")

    # Data rows
    for _, row in df.iterrows():
        row_str = " & ".join(str(v) for v in row.values) + " \\\\"
        latex_lines.append(row_str)

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    latex_lines.append("")

    return "\n".join(latex_lines)


def prettify_label(run_name: str) -> str:
    """Convert run name to pretty label."""
    # Remove _test suffix
    label = run_name.replace("_test", "")

    # Replace common abbreviations
    replacements = {
        "SFT_lr": "SFT lr=",
        "wSFT_lr": "wSFT lr=",
        "wSFT_vt": "wSFT vt=",
        "IRPO_beta": "IRPO β=",
        "DPO_beta": "DPO β=",
        "IRPO_lr": "IRPO lr=",
        "IRPO_w": "IRPO W=",
        "_l": " L=",
        "_alpha": " α=",
        "wSFT_deepseek_": "wSFT DeepSeek ",
        "wSFT_iter1_": "wSFT Iter1 ",
        "IRPO_deepseek_": "IRPO DeepSeek ",
        "IRPO_iter1_": "IRPO Iter1 ",
        "deepseek_": "DeepSeek ",
        "iter1_": "Iter1 ",
        "norep": "no replay",
        "rep0.": "replay=0.",
    }

    for old, new in replacements.items():
        label = label.replace(old, new)

    return label


def plot_group(data: Dict[str, pd.DataFrame], title: str, output_path: Path):
    """Generate improvement plot for a group of runs."""
    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))

    # Plot improvement with flipped sign (negative delta means improvement, show as positive)
    for i, (run_name, df) in enumerate(data.items()):
        pretty_name = prettify_label(run_name)

        improvement_flipped = -df["improvement"]
        ax.plot(
            df["n_value"],
            improvement_flipped,
            label=pretty_name,
            linewidth=2.5,
            color=colors[i],
            markersize=6,
            linestyle="-",
        )

    ax.set_xlabel("n (Best-of-N)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Improvement (avg -Δ)", fontsize=13, fontweight="bold")
    ax.set_xticks(range(1, 9))
    ax.grid(True, alpha=0.3, linestyle="--")

    # Title
    plt.title(title, fontsize=15, fontweight="bold", pad=20)

    # Legend
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output_path}")


def plot_group_split(data: Dict[str, pd.DataFrame], title: str, output_path: Path):
    """Generate split plot for second iteration (deepseek vs iter1 subplots)."""
    if not data:
        return

    # Split data by base model
    deepseek_data = {k: v for k, v in data.items() if "deepseek" in k}
    iter1_data = {k: v for k, v in data.items() if "iter1" in k}

    if not deepseek_data or not iter1_data:
        # Fall back to regular plot if split doesn't make sense
        plot_group(data, title, output_path)
        return

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))

    # Determine common y-axis range (only improvement)
    all_values = []
    for df in data.values():
        all_values.extend(-df["improvement"].values)  # Flipped improvement

    y_min = min(all_values) - 0.05
    y_max = max(all_values) + 0.05

    # Plot DeepSeek subplot
    colors_ds = plt.cm.tab10(np.linspace(0, 1, len(deepseek_data)))
    for i, (run_name, df) in enumerate(deepseek_data.items()):
        pretty_name = prettify_label(run_name)

        # Improvement with flipped sign
        improvement_flipped = -df["improvement"]
        ax_left.plot(
            df["n_value"],
            improvement_flipped,
            label=pretty_name,
            linewidth=2.5,
            color=colors_ds[i],
            markersize=6,
            linestyle="-",
        )

    ax_left.set_xlabel("n (Best-of-N)", fontsize=13, fontweight="bold")
    ax_left.set_ylabel("Improvement (avg -Δ)", fontsize=13, fontweight="bold")
    ax_left.set_xticks(range(1, 9))
    ax_left.set_ylim(y_min, y_max)
    ax_left.grid(True, alpha=0.3, linestyle="--")
    ax_left.set_title("DeepSeek Base", fontsize=14, fontweight="bold")
    ax_left.legend(loc="best", fontsize=9, framealpha=0.9)

    # Plot Iter1 subplot
    colors_i1 = plt.cm.tab10(np.linspace(0, 1, len(iter1_data)))
    for i, (run_name, df) in enumerate(iter1_data.items()):
        pretty_name = prettify_label(run_name)

        # Improvement with flipped sign
        improvement_flipped = -df["improvement"]
        ax_right.plot(
            df["n_value"],
            improvement_flipped,
            label=pretty_name,
            linewidth=2.5,
            color=colors_i1[i],
            markersize=6,
            linestyle="-",
        )

    ax_right.set_xlabel("n (Best-of-N)", fontsize=13, fontweight="bold")
    ax_right.set_ylabel("Improvement (avg -Δ)", fontsize=13, fontweight="bold")
    ax_right.set_xticks(range(1, 9))
    ax_right.set_ylim(y_min, y_max)
    ax_right.grid(True, alpha=0.3, linestyle="--")
    ax_right.set_title("Iter1 IRPO Base", fontsize=14, fontweight="bold")
    ax_right.legend(loc="best", fontsize=9, framealpha=0.9)

    # Overall title
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.00)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output_path}")


def generate_tables(all_runs: Dict[str, Dict[str, pd.DataFrame]]):
    """Generate all LaTeX tables."""
    print("\n" + "=" * 80)
    print("LATEX TABLES (n=8 metrics)")
    print("=" * 80 + "\n")

    # Table 1a: SFT/wSFT learning rate sweep
    if "wSFT_and_SFT_i1" in all_runs:
        lr_sweep, vt_sweep = categorize_first_iter_sft_wsft(all_runs["wSFT_and_SFT_i1"])

        if lr_sweep:
            lr_metrics = {name: extract_n8_metrics(df) for name, df in lr_sweep.items()}
            table = format_latex_table(
                lr_metrics,
                "First Iteration: SFT and wSFT Learning Rate Sweep",
                "tab:sft_wsft_lr",
            )
            print(table)

        if vt_sweep:
            vt_metrics = {name: extract_n8_metrics(df) for name, df in vt_sweep.items()}
            table = format_latex_table(
                vt_metrics,
                "First Iteration: wSFT Variance Threshold Sweep",
                "tab:wsft_vt",
            )
            print(table)

    # Tables 2a-c: IRPO/DPO
    if "IRPO_and_DPO_i1" in all_runs:
        beta_alpha, lr_sweep, wl_sweep = categorize_first_iter_irpo_dpo(
            all_runs["IRPO_and_DPO_i1"]
        )

        if beta_alpha:
            ba_metrics = {
                name: extract_n8_metrics(df) for name, df in beta_alpha.items()
            }
            table = format_latex_table(
                ba_metrics,
                "First Iteration: IRPO and DPO Beta/Alpha Sweep",
                "tab:irpo_dpo_beta_alpha",
            )
            print(table)

        if lr_sweep:
            lr_metrics = {name: extract_n8_metrics(df) for name, df in lr_sweep.items()}
            table = format_latex_table(
                lr_metrics, "First Iteration: IRPO Learning Rate Sweep", "tab:irpo_lr"
            )
            print(table)

        if wl_sweep:
            wl_metrics = {name: extract_n8_metrics(df) for name, df in wl_sweep.items()}
            table = format_latex_table(
                wl_metrics,
                "First Iteration: IRPO Winner/Loser Ratio Sweep",
                "tab:irpo_wl",
            )
            print(table)

    # Table 3: Second iteration straight IRPO
    if "straight_IRPO_i2" in all_runs:
        irpo_metrics = {
            name: extract_n8_metrics(df)
            for name, df in all_runs["straight_IRPO_i2"].items()
        }
        table = format_latex_table(
            irpo_metrics,
            "Second Iteration: Straight IRPO (Base Model and Replay Variations)",
            "tab:straight_irpo_i2",
        )
        print(table)

    # Tables 4a-b: Second iteration wSFT+IRPO
    if "wSFT_IRPO_i2" in all_runs:
        wsft_runs = {
            name: df
            for name, df in all_runs["wSFT_IRPO_i2"].items()
            if name.startswith("wSFT_")
        }
        irpo_runs = {
            name: df
            for name, df in all_runs["wSFT_IRPO_i2"].items()
            if name.startswith("IRPO_")
        }

        if wsft_runs:
            wsft_metrics = {
                name: extract_n8_metrics(df) for name, df in wsft_runs.items()
            }
            table = format_latex_table(
                wsft_metrics,
                "Second Iteration: wSFT Results (Base Model and Replay Variations)",
                "tab:wsft_i2",
            )
            print(table)

        if irpo_runs:
            irpo_metrics = {
                name: extract_n8_metrics(df) for name, df in irpo_runs.items()
            }
            table = format_latex_table(
                irpo_metrics,
                "Second Iteration: IRPO Results (Base Model and Replay Variations)",
                "tab:irpo_i2",
            )
            print(table)


def generate_plots(all_runs: Dict[str, Dict[str, pd.DataFrame]], output_dir: Path):
    """Generate all plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80 + "\n")

    # Plot 1a: SFT/wSFT learning rate sweep
    if "wSFT_and_SFT_i1" in all_runs:
        lr_sweep, vt_sweep = categorize_first_iter_sft_wsft(all_runs["wSFT_and_SFT_i1"])

        if lr_sweep:
            plot_group(
                lr_sweep,
                "SFT/wSFT Learning Rate Sweep",
                output_dir / "1a_sft_wsft_lr.png",
            )

        if vt_sweep:
            plot_group(
                vt_sweep, "wSFT Variance Threshold Sweep", output_dir / "1b_wsft_vt.png"
            )

    # Plots 2a-c: IRPO/DPO
    if "IRPO_and_DPO_i1" in all_runs:
        beta_alpha, lr_sweep, wl_sweep = categorize_first_iter_irpo_dpo(
            all_runs["IRPO_and_DPO_i1"]
        )

        if beta_alpha:
            plot_group(
                beta_alpha,
                "IRPO/DPO Beta/Alpha Sweep",
                output_dir / "2a_irpo_dpo_beta_alpha.png",
            )

        if lr_sweep:
            plot_group(
                lr_sweep, "IRPO Learning Rate Sweep", output_dir / "2b_irpo_lr.png"
            )

        if wl_sweep:
            plot_group(
                wl_sweep, "IRPO Winner/Loser Ratio Sweep", output_dir / "2c_irpo_wl.png"
            )

    # Plot 3: Second iteration straight IRPO
    if "straight_IRPO_i2" in all_runs:
        plot_group(
            all_runs["straight_IRPO_i2"],
            "Second Iteration: Straight IRPO",
            output_dir / "3_straight_irpo_i2.png",
        )

    # Plots 4a-b: Second iteration wSFT+IRPO (split by base model)
    if "wSFT_IRPO_i2" in all_runs:
        wsft_runs = {
            name: df
            for name, df in all_runs["wSFT_IRPO_i2"].items()
            if name.startswith("wSFT_")
        }
        irpo_runs = {
            name: df
            for name, df in all_runs["wSFT_IRPO_i2"].items()
            if name.startswith("IRPO_")
        }

        if wsft_runs:
            plot_group_split(
                wsft_runs, "Second Iteration: wSFT", output_dir / "4a_wsft_i2.png"
            )

        if irpo_runs:
            plot_group_split(
                irpo_runs, "Second Iteration: IRPO", output_dir / "4b_irpo_i2.png"
            )


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument(
        "--evals-dir",
        type=Path,
        default=Path("evals/ablations"),
        help="Path to evals/ablations directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ablations_analysis"),
        help="Directory for saving plots",
    )
    parser.add_argument(
        "--tables-only", action="store_true", help="Only generate tables, skip graphs"
    )
    parser.add_argument(
        "--graphs-only", action="store_true", help="Only generate graphs, skip tables"
    )

    args = parser.parse_args()

    # Load all run data
    print("Loading ablation run data...")
    all_runs = scan_ablation_runs(args.evals_dir)

    if not all_runs:
        print("No ablation runs found!")
        return 1

    # Generate tables
    if not args.graphs_only:
        generate_tables(all_runs)

    # Generate plots
    if not args.tables_only:
        generate_plots(all_runs, args.output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
