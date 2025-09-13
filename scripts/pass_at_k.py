import json
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict


# Color mapping for metrics
METRICS_COLOR_MAP = {
    "Resolved Instances": "#396ab1",  # blue
}


def calculate_pass_at_k_smooth(report_paths: List[str], N: int, fixed_first_run: bool = False, num_trials: int = 500) -> List[Tuple[float, float]]:
    """
    Calculate Pass@K rates with bootstrapping, adapted from the opt@k reference implementation.
    
    Args:
        report_paths: List of paths to JSON report files
        N: Maximum k value to compute pass@k for
        fixed_first_run: Whether to keep the first run performance unchanged
        num_trials: Number of bootstrap trials
        
    Returns:
        List of (mean, std) tuples for resolved instances
    """
    # Load all reports into a more usable format
    all_report_data = []
    for i, report_path in enumerate(report_paths):
        with open(report_path, "r") as f:
            report = json.load(f)
        all_report_data.append(report)

    # Create id_rankings_dict structure (instance_id -> run_id -> resolved)
    id_rankings_dict = {}
    for run_id, report in enumerate(all_report_data):
        # Extract resolved instance IDs
        resolved_ids = set(report.get("resolved_ids", []))
        
        # Get all instance IDs from the report
        all_instance_ids = set()
        all_instance_ids.update(resolved_ids)
        
        # Also check for other ID fields that might exist
        for key in report.keys():
            if key.endswith("_ids") and isinstance(report[key], list):
                all_instance_ids.update(report[key])

        for instance_id in all_instance_ids:
            if instance_id not in id_rankings_dict:
                id_rankings_dict[instance_id] = {}

            # Store resolved status
            id_rankings_dict[instance_id][run_id] = instance_id in resolved_ids

    total_instances = len(id_rankings_dict)
    resolved_at_n_trials = np.zeros((num_trials, N))

    # Run multiple trials
    for trial in range(num_trials):
        resolved_at_n = np.zeros(N)

        # Process each instance
        for instance_id, rankings in id_rankings_dict.items():
            rankings = list(rankings.items())  # (run_id, resolved)
            if not fixed_first_run:
                random.shuffle(rankings)

            # Process each N value
            for idx in range(N):
                # Shuffle for the second run (so we keep the first run perf unchanged)
                if fixed_first_run and idx == 1:
                    rankings = list(rankings)  # Make sure it's a list
                    random.shuffle(rankings)  # Shuffle for this trial

                n_rankings = rankings[: idx + 1]

                # Check if resolved in any
                if any(r[1] for r in n_rankings):
                    resolved_at_n[idx] += 1

        # Store results for this trial
        resolved_at_n_trials[trial] = resolved_at_n

    # Calculate means and standard deviations
    resolved_at_n_mean = np.mean(resolved_at_n_trials, axis=0)
    resolved_at_n_std = np.std(resolved_at_n_trials, axis=0)

    # Convert to percentages
    resolved_at_n_mean_pct = [x / total_instances * 100 for x in resolved_at_n_mean]
    resolved_at_n_std_pct = [x / total_instances * 100 for x in resolved_at_n_std]

    # Format results for plotting
    resolved_at_k_rates = list(zip(resolved_at_n_mean_pct, resolved_at_n_std_pct))

    return resolved_at_k_rates


def setup_plot_style():
    """Setup matplotlib style for clean, professional plots."""
    plt.rcParams.update(
        {
            "font.family": "Inter",
            "font.size": 15,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
        }
    )

    # Set cleaner grid style
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.linewidth"] = 0.5

    # Set cleaner axes style
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.linewidth"] = 1.0

    # Set figure DPI for better resolution
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300


def main():
    parser = argparse.ArgumentParser(description="Run and plot Pass@K evaluations")
    parser.add_argument(
        "--eval_reports", type=str, nargs="+", help="evaluated reports", required=True
    )
    parser.add_argument("--model_name", type=str, help="Model name", required=True)
    parser.add_argument("--k", type=int, default=10, help="Maximum K value")
    parser.add_argument(
        "--output_dir", type=str, default="plots", help="Directory to save plots"
    )
    parser.add_argument(
        "--fixed_first_run", action="store_true", help="Keep first run fixed across trials"
    )
    parser.add_argument(
        "--num_trials", type=int, default=1000, help="Number of bootstrap trials"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    reports = sorted(args.eval_reports)
    if len(reports) < args.k:
        raise ValueError(f"Found {len(reports)} reports, expected {args.k}")

    print(f"Found {len(reports)} report files:")
    for path in reports:
        print(f"  - {os.path.basename(path)}")

    print("\nCalculating Pass@K rates...")
    resolved_at_k_rates = calculate_pass_at_k_smooth(
        reports, args.k, args.fixed_first_run, args.num_trials
    )

    print(f"\nPass@K Results (N={args.k}):")
    print("k\tResolved")
    print("-" * 20)
    for i, resolved_rate in enumerate(resolved_at_k_rates):
        k = i + 1
        print(f"{k}\t{resolved_rate[0]:.1f}±{resolved_rate[1]:.1f}%")

    k_values = list(range(1, args.k + 1))
    plot_data = []

    for k, rates in enumerate(resolved_at_k_rates, 1):
        error = rates[1] if k < args.k else 0
        plot_data.append(
            {
                "k": k,
                "Rate": rates[0],
                "Error": error,
                "Metric": "Resolved Instances",
                "Lower": rates[0] - error,
                "Upper": rates[0] + error,
            }
        )

    df_plot = pd.DataFrame(plot_data)
    colors = METRICS_COLOR_MAP
    plt.figure(figsize=(6, 4), dpi=300)
    setup_plot_style()

    sns.lineplot(
        data=df_plot,
        x="k",
        y="Rate",
        hue="Metric",
        style="Metric",
        markers={"Resolved Instances": "o"},
        markeredgewidth=0,
        markersize=5,
        dashes=False,
        palette=colors,
    )

    for metric, color in colors.items():
        metric_data = df_plot[df_plot["Metric"] == metric]
        plt.fill_between(
            metric_data["k"],
            metric_data["Lower"],
            metric_data["Upper"],
            alpha=0.2,
            color=color,
            linewidth=0,
        )

    for metric, color in colors.items():
        metric_data = df_plot[df_plot["Metric"] == metric]
        for idx, (_, row) in enumerate(metric_data.iterrows()):
            if idx in [0, 1, 3, 5, 7, 9]:
                plt.annotate(
                    f'{row["Rate"]:.1f}',
                    (row["k"], row["Rate"]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    color="#3b3b3b",
                    fontsize=12,
                )

    plt.tick_params(axis="both", direction="out", length=3, width=1)
    plt.xlabel("# Rollouts (K)")
    plt.ylabel("Pass@K")
    plt.xticks(k_values)
    plt.ylim(50, 90)
    plt.grid(False)
    plt.legend(title=None, loc="lower right")

    output_path = os.path.join(args.output_dir, f"pass_at_k.{args.model_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {output_path}")


if __name__ == "__main__":
    main()
