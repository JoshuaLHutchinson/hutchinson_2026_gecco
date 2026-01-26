import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc("font", size=11)

# Constants and config
output_dir = "output"
envs = ["halfcheetah_uni", "hopper_uni", "walker2d_uni"]
operators = ["me_iso", "me_isoline", "me_isocross", "me_isolinecross"]
OPERATOR_LABELS = {
    "me_iso": "Iso",
    "me_isocross": "IsoCross",
    "me_isoline": "Iso+LineDD",
    "me_isolinecross": "IsoLineCross",
}

def load_ga_data(env_name, operator):
    folder = os.path.join(output_dir, env_name, operator)
    if not os.path.isdir(folder):
        return None

    dfs = []
    for run in sorted(os.listdir(folder))[:20]:
        run_path = os.path.join(folder, run, "log.csv")
        if os.path.isfile(run_path):
            df = pd.read_csv(run_path)
            dfs.append(df)
    if not dfs:
        return None
    return dfs

def plot_metric_over_time(env_name, metric):
    fig, ax = plt.subplots(figsize=(6, 4))
    for operator in operators:
        dfs = load_ga_data(env_name, operator)
        if dfs is None:
            continue

        # Aggregate mean and std over runs
        all_runs = []
        for df in dfs:
            if metric not in df.columns:
                continue
            all_runs.append(df[[metric, "iteration"]].set_index("iteration")[metric])
        if not all_runs:
            continue
        mean_df = pd.concat(all_runs, axis=1).mean(axis=1)
        std_df = pd.concat(all_runs, axis=1).std(axis=1)

        rolling_mean = mean_df.rolling(window=10, min_periods=1).mean()
        rolling_std = std_df.rolling(window=10, min_periods=1).mean()

        x_vals = mean_df.index.values * 256

        ax.plot(x_vals, rolling_mean.values, label=operator)
        ax.fill_between(x_vals, rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.25)

        if operator == "me_isolinecross":
            idx = np.linspace(0, len(x_vals)-1, 16, dtype=int)
            ax.scatter(x_vals[idx], rolling_mean.values[idx], s=18, color=ax.get_lines()[-1].get_color(), zorder=3)

    ax.set_xlabel("Evaluations (×10⁶)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.grid(axis="y", color="#e0e0e0", linewidth=0.8)
    ax.grid(axis="x", visible=False)

    ax.get_xaxis().get_offset_text().set_visible(False)

    # Legend below plot
    handles, labels = ax.get_legend_handles_labels()
    labels = [OPERATOR_LABELS.get(l, l) for l in labels]
    fig.legend(handles, labels, loc="lower center", ncol=len(operators), frameon=False)

    plt.tight_layout(pad=0.1, rect=[0, 0.05, 1, 1])
    output_path = os.path.join(output_dir, f"{env_name}_{metric}_ga_metrics.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.2)
    plt.show()

def plot_all_operators_mean_grid(metric):
    fig, axes = plt.subplots(nrows=len(envs), ncols=1, figsize=(6, 4*len(envs)), sharex=True)
    if len(envs) == 1:
        axes = [axes]
    for ax, env_name in zip(axes, envs):
        for operator in operators:
            dfs = load_ga_data(env_name, operator)
            if dfs is None:
                continue

            all_runs = []
            for df in dfs:
                if metric not in df.columns:
                    continue
                all_runs.append(df[[metric, "iteration"]].set_index("iteration")[metric])
            if not all_runs:
                continue
            mean_df = pd.concat(all_runs, axis=1).mean(axis=1)
            std_df = pd.concat(all_runs, axis=1).std(axis=1)

            rolling_mean = mean_df.rolling(window=10, min_periods=1).mean()
            rolling_std = std_df.rolling(window=10, min_periods=1).mean()

            x_vals = mean_df.index.values * 256

            ax.plot(x_vals, rolling_mean.values, label=operator)
            ax.fill_between(x_vals, rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.25)

            if operator == "me_isolinecross":
                idx = np.linspace(0, len(x_vals)-1, 16, dtype=int)
                ax.scatter(x_vals[idx], rolling_mean.values[idx], s=18, color=ax.get_lines()[-1].get_color(), zorder=3)

        ax.set_title(env_name.replace("_", " ").title())
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)

        ax.grid(axis="y", color="#e0e0e0", linewidth=0.8)
        ax.grid(axis="x", visible=False)

    axes[-1].set_xlabel("Evaluations (×10⁶)")
    axes[-1].get_xaxis().get_offset_text().set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    labels = [OPERATOR_LABELS.get(l, l) for l in labels]
    fig.legend(handles, labels, loc="lower center", ncol=len(operators), frameon=False)

    plt.tight_layout(pad=0.1, rect=[0, 0.05, 1, 1])
    output_path = os.path.join(output_dir, f"all_envs_{metric}_ga_metrics.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.2)
    plt.show()