import os
import pandas as pd
import yaml
from itertools import combinations
from scipy.stats import wilcoxon


# ---------------------------
# Config
# ---------------------------
output_folder = "output"
env_names = ["halfcheetah_uni", "walker2d_uni", "hopper_uni"]
metrics = ["qd_score", "coverage", "max_fitness"]
iteration_target = 4000
operators = ["me_iso", "me_isoline", "me_isocross", "me_isolinecross"]
MAX_RUNS_PER_OPERATOR = 20  # set None for all runs

# ---------------------------
# Load data (all envs)
# ---------------------------
all_results = []

for env_name in env_names:
    for op in operators:
        op_folder = os.path.join(output_folder, env_name, op)
        if not os.path.isdir(op_folder):
            continue

        ts_folders = [
            d for d in os.listdir(op_folder)
            if os.path.isdir(os.path.join(op_folder, d))
        ]
        ts_folders = sorted(ts_folders)

        if MAX_RUNS_PER_OPERATOR is not None:
            ts_folders = ts_folders[:MAX_RUNS_PER_OPERATOR]

        for ts_folder in ts_folders:
            ts_path = os.path.join(op_folder, ts_folder)

            log_csv_path = os.path.join(ts_path, "log.csv")
            if not os.path.isfile(log_csv_path):
                continue

            df = pd.read_csv(log_csv_path)
            row = df[df["iteration"] == iteration_target]
            if row.empty:
                continue

            config = {}
            config_path = os.path.join(ts_path, ".hydra", "config.yaml")
            if os.path.isfile(config_path):
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)

            for metric_name in metrics:
                all_results.append(
                    {
                        "env": env_name,
                        "operator": op,
                        "seed": config.get("seed", None),
                        "metric": metric_name,
                        "value": row[metric_name].values[0],
                    }
                )

results_df = pd.DataFrame(all_results)

# Drop entries without seeds (cannot be paired)
results_df = results_df.dropna(subset=["seed"])

# ---------------------------
# Summary statistics
# ---------------------------
summary = (
    results_df
    .groupby(["env", "operator", "metric"])["value"]
    .agg(["count", "mean", "std"])
    .reset_index()
)

summary.rename(columns={"count": "n_runs"}, inplace=True)

print("\nMean and std at iteration", iteration_target)
print(summary.to_string(index=False, float_format="%.4f"))

# ---------------------------
# Paired Wilcoxon tests
# (percentage mean increase, targeted comparisons)
# ---------------------------
wilcoxon_results = []

# Define targeted comparisons: (baseline, comparison)
target_pairs = [
    ("me_isoline", "me_isolinecross"),
    ("me_isoline", "me_isocross"),
    ("me_isocross", "me_isolinecross"),
]

for env_name in env_names:
    for metric_name in metrics:
        env_metric_df = results_df[
            (results_df["env"] == env_name) &
            (results_df["metric"] == metric_name)
        ]

        # Pivot so rows = seeds, columns = operators
        pivot = env_metric_df.pivot(
            index="seed",
            columns="operator",
            values="value"
        )

        for baseline_op, compare_op in target_pairs:
            if baseline_op not in pivot.columns or compare_op not in pivot.columns:
                continue

            paired = pivot[[baseline_op, compare_op]].dropna()
            if len(paired) < 2:
                continue  # Wilcoxon requires at least 2 paired samples

            baseline_vals = paired[baseline_op]
            compare_vals = paired[compare_op]

            # Percentage mean increase: ((B - A) / A) * 100
            pct_mean_increase = ((compare_vals - baseline_vals) / baseline_vals).mean() * 100.0

            try:
                stat, p_value = wilcoxon(baseline_vals, compare_vals)
            except ValueError:
                continue  # e.g. all differences zero

            wilcoxon_results.append(
                {
                    "env": env_name,
                    "metric": metric_name,
                    "baseline_operator": baseline_op,
                    "comparison_operator": compare_op,
                    "n_pairs": len(paired),
                    "mean_pct_increase": pct_mean_increase,
                    "p_value": p_value,
                }
            )

wilcoxon_df = pd.DataFrame(wilcoxon_results)

print("\nPaired Wilcoxon results (percentage mean increase vs isoline)")
print(
    wilcoxon_df.to_string(
        index=False,
        float_format="%.4f"
    )
)