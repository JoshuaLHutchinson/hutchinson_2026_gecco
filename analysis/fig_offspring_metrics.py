import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc("font", size=11)

OPERATORS = [
    "me_isolinecross",
    "me_isoline",
    "me_isocross",
    "me_iso",
]

OPERATOR_LABELS = {
    "me_isolinecross": "IsoLineCross",
    "me_isoline": "Iso+LineDD",
    "me_isocross": "IsoCross",
    "me_iso": "Iso",
}


ROLLING_WINDOW = 50

ENVS = [
    "halfcheetah_uni",
    "hopper_uni",
    "walker2d_uni",
]

ENV_DISPLAY_NAMES = {
    "halfcheetah_uni": "HalfCheetah Uni",
    "hopper_uni": "Hopper Uni",
    "walker2d_uni": "Walker2d Uni",
}


def load_all_runs(
    output_dir: Path,
    env_name: str,
    operator: str,
    metric: str,
):
    """
    Load log.csv from all timestamp folders for one operator.
    Returns a list of DataFrames with ['iteration', metric].
    """
    operator_dir = output_dir / env_name / operator
    dfs = []

    if not operator_dir.exists():
        return dfs

    for timestamp_dir in operator_dir.iterdir():
        if not timestamp_dir.is_dir():
            continue

        log_csv = timestamp_dir / "log.csv"
        if not log_csv.exists():
            continue

        df = pd.read_csv(log_csv)

        if "iteration" not in df.columns or metric not in df.columns:
            continue

        dfs.append(df[["iteration", metric]])

    return dfs


def compute_mean_over_runs(dfs):
    """
    Concatenate runs and compute mean per iteration.
    """
    if not dfs:
        return None

    all_runs = pd.concat(dfs, axis=0)
    mean_df = (
        all_runs.groupby("iteration", as_index=False).mean().sort_values("iteration")
    )
    return mean_df


def plot_metric_over_time(
    output_dir: Path,
    env_name: str,
    metric: str,
):
    fig, ax = plt.subplots(figsize=(9, 5))

    for operator in OPERATORS:
        dfs = load_all_runs(
            output_dir=output_dir,
            env_name=env_name,
            operator=operator,
            metric=metric,
        )

        mean_df = compute_mean_over_runs(dfs)
        if mean_df is None:
            continue

        rolling_mean = (
            mean_df[metric].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )

        x_vals = mean_df["iteration"].values
        ax.plot(
            x_vals,
            rolling_mean,
            label=OPERATOR_LABELS[operator],
        )
        if operator == "me_isolinecross":
            idx = np.linspace(0, len(x_vals)-1, 16, dtype=int)
            ax.scatter(x_vals[idx], rolling_mean.values[idx], s=18, color=ax.get_lines()[-1].get_color(), zorder=3)

    ax.set_xlabel("Generation")
    ax.set_ylabel(metric)
    ax.set_title(f"Mean {metric} per iteration")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.legend(loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.3))

    plt.tight_layout(pad=0.1, rect=[0, 0.05, 1, 1])
    output_path = os.path.join(output_dir, f"{env_name}_ga_metrics.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.2)
    plt.show()


def load_all_runs_with_seed(
    output_dir: Path,
    env_name: str,
    operator: str,
    metric: str,
):
    """
    Load log.csv from all timestamp folders for one operator.
    Returns a list of DataFrames with ['iteration', metric, 'seed'].
    Assumes seed is stored in .hydra/config/config.yaml in each run folder.
    """
    import yaml

    operator_dir = output_dir / env_name / operator
    dfs = []

    if not operator_dir.exists():
        return dfs

    for timestamp_dir in operator_dir.iterdir():
        if not timestamp_dir.is_dir():
            continue

        log_csv = timestamp_dir / "log.csv"
        config_yaml = timestamp_dir / ".hydra" / "config.yaml"
        if not log_csv.exists() or not config_yaml.exists():
            continue

        with open(config_yaml, "r") as f:
            cfg = yaml.safe_load(f)
            seed = cfg.get("seed", None)

        if seed is None:
            continue

        df = pd.read_csv(log_csv)
        if "iteration" not in df.columns or metric not in df.columns:
            continue

        df["seed"] = seed
        dfs.append(df[["iteration", metric, "seed"]])

    return dfs


def plot_all_operators_mean_grid(
    output_dir: Path,
    rolling_window: int = 10,
    start_gen: int = 0,
):
    """
    Plots a grid:
    Rows = environments (ENVS)
    Columns = metrics:
        1) ga_offspring_added
        2) ga_improvement
        3) ga_improvement / ga_offspring_added

    For each panel, only the mean across seeds is shown (no background seed-wise runs).
    Operators are plotted in the order defined in OPERATORS (ranking).
    """

    metrics = ["ga_offspring_added", "ga_improvement"]

    fig, axes = plt.subplots(
        nrows=len(ENVS),
        ncols=2,
        figsize=(6, 8),
        sharex="col",
    )

    for row_idx, env_name in enumerate(ENVS):
        for op_idx, operator in enumerate(OPERATORS):
            # Load per-operator data
            dfs = {
                m: pd.concat(
                    load_all_runs_with_seed(output_dir, env_name, operator, m),
                    ignore_index=True,
                )
                for m in metrics
            }

            if any(df.empty for df in dfs.values()):
                continue

            # Rolling mean per seed
            for m in metrics:
                dfs[m][f"roll_{m.split('_')[-1]}"] = (
                    dfs[m]
                    .groupby("seed")[m]
                    .transform(
                        lambda x: x.rolling(rolling_window, min_periods=1).mean()
                    )
                )
                dfs[m] = dfs[m][dfs[m]["iteration"] >= start_gen]

            stats_off = (
                dfs["ga_offspring_added"]
                .groupby("iteration")[f"roll_added"]
                .agg(["mean", "std"])
                .reset_index()
            )

            ratio_seed = pd.merge(
                dfs["ga_improvement"],
                dfs["ga_offspring_added"],
                on=["seed", "iteration"],
                suffixes=("_imp", "_off"),
            )

            ratio_seed["ratio"] = ratio_seed["roll_improvement"] / ratio_seed["roll_added"]

            ratio = (
                ratio_seed
                .groupby("iteration")["ratio"]
                .agg(["mean", "std"])
                .reset_index()
            )

            panels = [
                (stats_off, "Offspring Added"),
                (ratio, "QD Score Added per Offspring"),
            ]

            for col_idx, (df, title) in enumerate(panels):
                ax = axes[row_idx, col_idx]

                x_vals = df["iteration"].values
                ax.plot(
                    x_vals,
                    df["mean"],
                    linewidth=1.5,
                    label=OPERATOR_LABELS[operator] if row_idx == 0 else None,
                )
                if operator == "me_isolinecross":
                    idx = np.linspace(0, len(x_vals)-1, 16, dtype=int)
                    ax.scatter(x_vals[idx], df["mean"].values[idx], s=18, color=ax.get_lines()[-1].get_color(), zorder=3)


                if row_idx == len(ENVS) - 1:
                    ax.set_xlabel("Generation")

                if row_idx == 0:
                    ax.set_title(title)

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(True)
                ax.spines["bottom"].set_visible(True)

            axes[row_idx, 0].set_ylabel(ENV_DISPLAY_NAMES[env_name])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    plt.tight_layout(pad=0.1, rect=[0, 0.05, 1, 1])
    output_path = os.path.join(output_dir, "all_envs_ga_metrics.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.2)
    plt.show()


def print_env_summary_table(output_dir: Path, window_size: int = 500):
    """
    Prints a table of percentage change in ga_improvement and ga_offspring_added
    of me_isolinecross over me_isoline per environment for each window of `window_size` iterations.
    Also prints the overall mean QD score added per offspring for IsoLineCross and IsoLine, aggregated across all environments for generations >= 1000.
    """
    metrics = ["ga_offspring_added", "ga_improvement"]

    print("\nSummary table of % change in GA metrics (IsoLineCross over IsoLine) per environment per window:")
    totals = {
        "0-1000": {"cross_imp": 0.0, "cross_off": 0.0, "line_imp": 0.0, "line_off": 0.0},
        "1000+": {"cross_imp": 0.0, "cross_off": 0.0, "line_imp": 0.0, "line_off": 0.0},
        "all": {"cross_imp": 0.0, "cross_off": 0.0, "line_imp": 0.0, "line_off": 0.0},
    }
    for env_name in ENVS:
        print(f"\nEnvironment: {env_name}")

        # Load data for operator pair
        dfs_isolinecross = {
            m: pd.concat(
                load_all_runs_with_seed(output_dir, env_name, "me_isolinecross", m),
                ignore_index=True,
            )
            for m in metrics
        }
        dfs_isoline = {
            m: pd.concat(
                load_all_runs_with_seed(output_dir, env_name, "me_isoline", m),
                ignore_index=True,
            )
            for m in metrics
        }

        # Check for missing data
        if any(df.empty for df in dfs_isolinecross.values()) or any(
            df.empty for df in dfs_isoline.values()
        ):
            print("  Missing runs for one or more operators. Skipping.")
            continue

        # Merge data for the pair
        merged_linecross_line = {}

        for m in metrics:
            merged_linecross_line[m] = pd.merge(
                dfs_isolinecross[m],
                dfs_isoline[m],
                on=["seed", "iteration"],
                suffixes=("_cross", "_line"),
            )

        iters = merged_linecross_line["ga_improvement"]["iteration"]

        masks = {
            "0-1000": iters < 1000,
            "1000+": iters >= 1000,
            "all": iters >= 0,
        }

        for key, mask in masks.items():
            totals[key]["cross_imp"] += merged_linecross_line["ga_improvement"].loc[
                mask, "ga_improvement_cross"
            ].sum()
            totals[key]["cross_off"] += merged_linecross_line["ga_offspring_added"].loc[
                mask, "ga_offspring_added_cross"
            ].sum()
            totals[key]["line_imp"] += merged_linecross_line["ga_improvement"].loc[
                mask, "ga_improvement_line"
            ].sum()
            totals[key]["line_off"] += merged_linecross_line["ga_offspring_added"].loc[
                mask, "ga_offspring_added_line"
            ].sum()

        max_iter = int(
            max(
                merged_linecross_line["ga_offspring_added"]["iteration"].max(),
                merged_linecross_line["ga_improvement"]["iteration"].max(),
            )
        )
        windows = [
            (start, min(start + window_size - 1, max_iter))
            for start in range(1, max_iter + 1, window_size)
        ]

        print(
            f"{'Window':<12} {'% Change Offspring Added':<30} {'% Change Improvement':<30}"
        )
        for start, end in windows:
            mask_line = (merged_linecross_line["ga_offspring_added"]["iteration"] >= start) & (
                merged_linecross_line["ga_offspring_added"]["iteration"] <= end
            )

            mean_cross_offspring = merged_linecross_line["ga_offspring_added"].loc[mask_line, "ga_offspring_added_cross"].mean()
            mean_line_offspring = merged_linecross_line["ga_offspring_added"].loc[mask_line, "ga_offspring_added_line"].mean()

            # Compute improvement per offspring as sums of improvement divided by sums of offspring
            sum_cross_improvement = merged_linecross_line["ga_improvement"].loc[mask_line, "ga_improvement_cross"].sum()
            sum_cross_offspring = merged_linecross_line["ga_offspring_added"].loc[mask_line, "ga_offspring_added_cross"].sum()
            sum_line_improvement = merged_linecross_line["ga_improvement"].loc[mask_line, "ga_improvement_line"].sum()
            sum_line_offspring = merged_linecross_line["ga_offspring_added"].loc[mask_line, "ga_offspring_added_line"].sum()

            mean_cross_improvement = sum_cross_improvement / sum_cross_offspring if sum_cross_offspring != 0 else float('nan')
            mean_line_improvement = sum_line_improvement / sum_line_offspring if sum_line_offspring != 0 else float('nan')

            # Compute percentage change safely (avoid division by zero)
            perc_change_offspring = ((mean_cross_offspring - mean_line_offspring) / mean_line_offspring * 100) if mean_line_offspring != 0 else float('nan')
            perc_change_improvement = ((mean_cross_improvement - mean_line_improvement) / mean_line_improvement * 100) if mean_line_improvement != 0 else float('nan')

            print(f"{start}-{end:<10} {perc_change_offspring:<30.2f} {perc_change_improvement:<30.2f}")

    print("\nOverall mean QD score added per offspring by generation range")
    for key, vals in totals.items():
        print(f"\nGenerations {key}:")
        if vals["cross_off"] > 0 and vals["line_off"] > 0:
            mean_cross_qd = vals["cross_imp"] / vals["cross_off"]
            mean_line_qd = vals["line_imp"] / vals["line_off"]
            perc_increase = (mean_cross_qd - mean_line_qd) / mean_line_qd * 100

            print(f"  IsoLineCross: {mean_cross_qd:.6f}")
            print(f"  IsoLine:      {mean_line_qd:.6f}")
            print(f"  % increase:   {perc_increase:.2f}%")
        else:
            print("  Insufficient data to compute statistics.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    plot_all_operators_mean_grid(
        output_dir=args.output_dir,
        rolling_window=ROLLING_WINDOW,
        start_gen=1000,
    )

    print_env_summary_table(
        output_dir=args.output_dir,
        window_size=500,
    )


if __name__ == "__main__":
    main()
