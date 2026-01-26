import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from cross_mutation_operators import isolinecross_variation

# ==========================
# PARAMETERS (fill in as needed)
# ==========================
N = 10000  # number of simulations per setting
iso_sigma = 0.02
line_sigma = 0.15
cross_lambda = 2
cross_prob = 0.5
minval = None
maxval = None

# Parent genotypes (2D for plotting)
p1 = jnp.array([[0.3, 0.3]])
p2 = jnp.array([[0.7, 0.7]])

# Settings to simulate
settings = ["iso", "isocross", "isoline", "isolinecross"]

# Random key
key = jax.random.PRNGKey(0)

# ==========================
# FUNCTION TO RUN VARIATION
# ==========================
def run_variation(setting, x1, x2, key):
    if setting == "iso":
        x_off, key = isolinecross_variation(
            x1, x2, key,
            iso_sigma=iso_sigma,
            line_sigma=0.0,
            cross_lambda=0.0,
            cross_prob=0.0,
            minval=minval,
            maxval=maxval,
        )
    elif setting == "isocross":
        x_off, key = isolinecross_variation(
            x1, x2, key,
            iso_sigma=iso_sigma,
            line_sigma=0.0,
            cross_lambda=cross_lambda,
            cross_prob=cross_prob,
            minval=minval,
            maxval=maxval,
        )
    elif setting == "isoline":
        x_off, key = isolinecross_variation(
            x1, x2, key,
            iso_sigma=iso_sigma,
            line_sigma=line_sigma,
            cross_lambda=0.0,
            cross_prob=0.0,
            minval=minval,
            maxval=maxval,
        )
    elif setting == "isolinecross":
        x_off, key = isolinecross_variation(
            x1, x2, key,
            iso_sigma=iso_sigma,
            line_sigma=line_sigma,
            cross_lambda=cross_lambda,
            cross_prob=cross_prob,
            minval=minval,
            maxval=maxval,
        )
    else:
        raise ValueError(f"Unknown setting {setting}")
    return x_off, key

# ==========================
# SIMULATE AND PLOT
# ==========================
fig, axes = plt.subplots(1, 4, figsize=(8, 2))
labels = ['(a) Iso', '(b) IsoCross', '(c) Iso+LineDD', '(d) IsoLineCross']
for i, setting in enumerate(settings):
    offspring_points = []
    key_sim = key
    for j in range(N):
        if j < N // 2:
            x_off, key_sim = run_variation(setting, p1, p2, key_sim)
        else:
            x_off, key_sim = run_variation(setting, p2, p1, key_sim)
        offspring_points.append(x_off[0])  # extract 2D point

    offspring_points = jnp.array(offspring_points)
    ax = axes[i]
    # plot offspring
    ax.scatter(offspring_points[:, 0], offspring_points[:, 1], color='red', alpha=0.5, label='Offspring', s=0.5)
    # plot parents
    ax.scatter(p1[:, 0], p1[:, 1], color='black', label='Parent 1', s=20)
    ax.scatter(p2[:, 0], p2[:, 1], color='black', label='Parent 2', s=20)
    # plot parents on top
    label = labels[i]
    ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=10, va='top')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()