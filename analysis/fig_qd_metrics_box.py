import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc("font", size=11)

# ---------------------------
# Config
# ---------------------------
output_folder = "output"
envs = ["halfcheetah_uni", "hopper_uni", "walker2d_uni"]
env_display_names = {
    "halfcheetah_uni": "HalfCheetah Uni",
    "walker2d_uni": "Walker2d Uni",
    "hopper_uni": "Hopper Uni",
}
operators = ["me_iso", "me_isoline", "me_isocross", "me_isolinecross"]
metrics = ["qd_score", "coverage", "max_fitness"]
metric_titles = ["QD Score", "Coverage (%)", "Max Fitness"]

MAX_RUNS_PER_OPERATOR = 20  # None for all
GRID_COLOR = "#e0e0e0"

# ---------------------------
# Helper
# ---------------------------
def load_runs(env, operator, metric):
    op_folder = os.path.join(output_folder, env, operator)
    if not os.path.isdir(op_folder):
        return {}

    ts_folders = sorted(
        d for d in os.listdir(op_folder)
        if os.path.isdir(os.path.join(op_folder, d))
    )
    if MAX_RUNS_PER_OPERATOR is not None:
        ts_folders = ts_folders[:MAX_RUNS_PER_OPERATOR]

    runs = []
    for ts in ts_folders:
        log_path = os.path.join(op_folder, ts, "log.csv")
        if not os.path.isfile(log_path):
            continue
        df = pd.read_csv(log_path)
        if "evaluation" not in df.columns:
            df["evaluation"] = df["iteration"]
        runs.append(df[["evaluation", metric]].rename(columns={metric: "value"}))
    return runs


def aggregate_runs(runs):
    if not runs:
        return None
    df = pd.concat(
        [r.assign(run=i) for i, r in enumerate(runs)],
        ignore_index=True
    )
    grouped = df.groupby("evaluation")["value"]
    return pd.DataFrame({
        "median": grouped.median(),
        "q1": grouped.quantile(0.25),
        "q3": grouped.quantile(0.75),
    }).reset_index()


# ---------------------------
# Rank operators by final QD score (averaged over envs)
# ---------------------------
final_scores = {}
for op in operators:
    vals = []
    for env in envs:
        runs = load_runs(env, op, "qd_score")
        agg = aggregate_runs(runs)
        if agg is not None:
            vals.append(agg["median"].iloc[-1])
    final_scores[op] = np.mean(vals) if vals else -np.inf

ranked_ops = sorted(final_scores, key=final_scores.get, reverse=True)

# ---------------------------
# Plot box plots
# ---------------------------
fig, axes = plt.subplots(
    nrows=len(envs),
    ncols=len(metrics),
    figsize=(12, 7.5),
    sharey=False
)

for row, env in enumerate(envs):
    for col, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[row, col]

        box_data = []
        labels = []
        for op in ranked_ops:
            runs = load_runs(env, op, metric)
            if not runs:
                continue
            # Take final evaluation for each run
            final_vals = [r["value"].iloc[-1] for r in runs]
            if metric == "qd_score":
                final_vals = [v / 1e5 for v in final_vals]
            box_data.append(final_vals)
            labels.append(op)

        if box_data:
            bp = ax.boxplot(
                box_data,
                vert=True,
                patch_artist=True,
                showmeans=False,
                showfliers=True,
                labels=["" for _ in labels]  # hide x labels
            )
            
            for median in bp['medians']:
                median.set(color='black', linewidth=1)
            
            colors = plt.cm.tab10.colors
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

        ax.set_title(title if row == 0 else "")
        if col == 0:
            ax.set_ylabel(env_display_names[env])
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.grid(axis="x", visible=False)

        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

        if metric == "qd_score" and col == 0:
            ax.text(
                -0.08, 1.02, "×10⁵", transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='left', fontsize=10
            )

# Custom legend
legend_label_map = {
    "me_iso": "Iso",
    "me_isocross": "IsoCross",
    "me_isoline": "Iso+LineDD",
    "me_isolinecross": "IsoLineCross",
}

handles = [plt.Line2D([0], [0], color=plt.cm.tab10(i), lw=4) for i in range(len(ranked_ops))]
labels = [legend_label_map[l] for l in ranked_ops]
fig.legend(handles, labels, loc="lower center", ncol=len(ranked_ops), frameon=False)

plt.tight_layout(pad=0.1, h_pad=1.0, rect=[0, 0.05, 1, 1])
output_path = os.path.join(output_folder, "reproducibility_box.pdf")
plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()