from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import jax.numpy as jnp
from qdax.utils.plotting import get_voronoi_finite_polygons_2d

from utils import get_config, get_env, get_repertoire


# Define env and algo names
ENV = "halfcheetah_uni"

ENV_LIST = [
    "walker2d_uni",
    "halfcheetah_uni",
]
ENV_DICT = {
    "walker2d_uni": "Walker Uni",
    "halfcheetah_uni": "HalfCheetah Uni",
}
DESC_DICT = {
    "walker2d_uni": ["Foot Contact - Leg 1", "Foot Contact - Leg 2"],
    "halfcheetah_uni": ["Foot Contact - Leg 1", "Foot Contact - Leg 2"],
}


# Use operators from fig_qd_metrics.py
OPERATORS = ["me_iso", "me_isocross", "me_isoline", "me_isolinecross"]
OPERATOR_LABELS = {
    "me_iso": "Iso",
    "me_isocross": "IsoCross",
    "me_isoline": "Iso+LineDD",
    "me_isolinecross": "IsoLineCross",
}


def get_df(results_dir: Path):
    dfs = []
    for env_dir in results_dir.iterdir():
        if not env_dir.is_dir():
            continue
        env = env_dir.name
        for operator_dir in env_dir.iterdir():
            if not operator_dir.is_dir():
                continue
            operator = operator_dir.name
            for run_dir in operator_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                run = run_dir.name
                csv_path = run_dir / "log.csv"
                if csv_path.is_file():
                    df = pd.read_csv(csv_path)
                    df["env"] = env
                    df["operator"] = operator
                    df["run"] = run
                    dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No CSV files found in {results_dir}")
    return pd.concat(dfs, ignore_index=True)


def plot_2d_repertoire(
    ax, repertoire, minval, maxval, vmin, vmax, display_descriptors=False, cbar=False
):
    """Plot a 2d map elites repertoire on the given axis."""
    assert repertoire.centroids.shape[-1] == 2, "Descriptor space must be 2d"

    repertoire_empty = repertoire.fitnesses == -jnp.inf

    # Set axes limits
    ax.set_xlim(minval[0], maxval[0])
    ax.set_ylim(minval[1], maxval[1])
    ax.set(adjustable="box", aspect="equal")

    # Create the regions and vertices from centroids
    regions, vertices = get_voronoi_finite_polygons_2d(repertoire.centroids)

    # Colors
    cmap = matplotlib.cm.viridis
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Fill the plot with contours
    for region in regions:
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1)

    # Fill the plot with the colors
    for idx, fitness in enumerate(repertoire.fitnesses):
        if fitness > -jnp.inf:
            region = regions[idx]
            polygon = vertices[region]
            ax.fill(*zip(*polygon), alpha=0.8, color=cmap(norm(fitness)))

    # if descriptors are specified, add points location
    if display_descriptors:
        descriptors = repertoire.descriptors[~repertoire_empty]
        ax.scatter(
            descriptors[:, 0],
            descriptors[:, 1],
            c=repertoire.fitnesses[~repertoire_empty],
            cmap=cmap,
            s=10,
            zorder=0,
        )

    # Aesthetic
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cbar.ax.tick_params()
    ax.set_aspect("equal")

    return ax


def plot(repertoires, configs):
    # Check that all configs have the same env
    envs = [config.env.name for config in configs]
    assert envs.count(envs[0]) == len(envs), "All configs must have the same env"

    # Get env to get descriptor space limits
    env = get_env(configs[0])

    # Get max and min fitness for each repertoire
    min_list = [get_min_fitnesses(repertoire) for repertoire in repertoires]
    max_list = [get_max_fitnesses(repertoire) for repertoire in repertoires]

    # Get vmin and vmax
    vmin = min(min_list)
    vmax = max(max_list)

    # Get minval and maxval
    minval, maxval = env.behavior_descriptor_limits

    # Create subplots
    fig, axes = plt.subplots(
        nrows=1, ncols=len(repertoires), sharey=True, squeeze=False, figsize=(25, 5)
    )

    # Plot each repertoire
    for col, (repertoire, config) in enumerate(zip(repertoires, configs)):
        # Set title for the column
        axes[0, col].set_title(
            OPERATOR_LABELS.get(config.operator.name, config.operator.name)
        )

        # Set the x and y labels
        axes[0, col].set_xlabel(DESC_DICT[envs[0]][0])
        if col == 0:
            axes[0, col].set_ylabel(DESC_DICT[envs[0]][1])
        else:
            axes[0, col].set_ylabel(None)

        # Plot repertoire
        plot_2d_repertoire(axes[0, col], repertoire, minval, maxval, vmin, vmax)

    # cax = fig.add_axes([0.92, 0.1, 0.01, 0.8])
    cax = fig.add_axes([0.91, 0.1, 0.01, 0.754])
    cmap = matplotlib.cm.viridis
    norm = Normalize(vmin=vmin, vmax=vmax)
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

    return fig


def get_min_fitnesses(repertoire):
    return repertoire.fitnesses.min(
        initial=jnp.inf, where=(repertoire.fitnesses != -jnp.inf)
    )


def get_max_fitnesses(repertoire):
    return repertoire.fitnesses.max(
        initial=-jnp.inf, where=(repertoire.fitnesses != -jnp.inf)
    )


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rc("font", size=14)

    # Create the DataFrame
    results_dir = Path("output/")
    df = get_df(results_dir)

    # Filter
    df = df[df["operator"].isin(OPERATORS)]

    # Get the median QD score for each (env, operator)
    idx = df.groupby(["env", "operator", "run"])["iteration"].idxmax()
    df_last_iteration = df.loc[idx]
    qd_score_median = df_last_iteration.groupby(["env", "operator"])[
        "qd_score"
    ].median()
    df_last_iteration = df_last_iteration.join(
        qd_score_median, on=["env", "operator"], rsuffix="_median"
    )

    # Get the difference between the QD score and the median for each run
    df_last_iteration["qd_score_diff_to_median"] = abs(
        df_last_iteration["qd_score"] - df_last_iteration["qd_score_median"]
    )

    # Get the most representative run for each (env, operator)
    idx = df_last_iteration.groupby(["env", "operator"])[
        "qd_score_diff_to_median"
    ].idxmin()
    runs = df_last_iteration.loc[idx][["env", "operator", "run"]]

    # Get run paths
    run_paths = [
        results_dir
        / ENV
        / operator
        / runs[(runs["env"] == ENV) & (runs["operator"] == operator)]["run"].item()
        for operator in OPERATORS
    ]

    # Get configs
    configs = [get_config(run_path) for run_path in run_paths]

    # Get repertoires
    repertoires = [get_repertoire(run_path) for run_path in run_paths]

    # Plot
    fig = plot(repertoires, configs)
    fig.savefig(f"plot_archive_{ENV}.pdf", bbox_inches="tight")
    plt.close()
