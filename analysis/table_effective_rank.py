import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
FINAL_BASE_DIR = Path("output")
INITIAL_BASE_DIR = Path("output_inital_population")

ENVS = [
    #"halfcheetah_uni",
    #"hopper_uni",
    "walker2d_uni",
]

OPERATORS = [
    "me_iso",
    "me_isocross",
    "me_isoline",
    "me_isolinecross",
]

GENOTYPE_FILE = "repertoire/genotypes.npy"
EIG_TOL = 1e-12  # numerical stability threshold
VARIANCE_THRESHOLD = 0.95  # for effective rank


# -----------------------------
# Core computation
# -----------------------------
def effective_rank_and_dims(X: np.ndarray, variance_threshold: float = 0.95):
    """
    Returns:
      - effective rank k
      - set of dimension indices 'used' by the top-k principal components

    Uses SVD on centered data so principal directions are expressed
    in the original parameter space (dimensions are interpretable).
    """
    # Center
    Xc = X - X.mean(axis=0, keepdims=True)

    # SVD
    # Xc = U S V^T, rows of V^T are principal directions
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    # Variance explained
    eigvals = (S ** 2) / (len(Xc) - 1)
    var_explained = np.cumsum(eigvals) / np.sum(eigvals)

    k = int(np.searchsorted(var_explained, variance_threshold)) + 1

    # Dimensions "used": for each of top-k PCs,
    # take the dimension with largest absolute loading
    used_dims = set(np.argmax(np.abs(Vt[:k]), axis=1))

    return k, used_dims


# -----------------------------
# Data loading
# -----------------------------
records = []

for env in ENVS:
    for operator in OPERATORS:
        init_runs = sorted((INITIAL_BASE_DIR / env / operator).iterdir())
        final_runs = sorted((FINAL_BASE_DIR / env / operator).iterdir())

        if len(init_runs) != len(final_runs):
            raise RuntimeError(
                f"Seed count mismatch for {env} / {operator}: "
                f"{len(init_runs)} initial vs {len(final_runs)} final"
            )

        for seed_idx, (init_run, final_run) in enumerate(zip(init_runs, final_runs)):
            try:
                X_init = np.load(init_run / "repertoire/genotypes.npy")
                X_final = np.load(final_run / "repertoire/genotypes.npy")

                k_init, dims_init = effective_rank_and_dims(
                    X_init, VARIANCE_THRESHOLD
                )
                k_final, dims_final = effective_rank_and_dims(
                    X_final, VARIANCE_THRESHOLD
                )

                # Overlap metrics
                intersection = dims_init & dims_final

                # Jaccard overlap
                if len(dims_init | dims_final) == 0:
                    overlap = 0.0
                else:
                    overlap = len(intersection) / len(dims_init | dims_final)

                # Directional overlaps
                # % of final dimensions already present in initial
                pct_final_in_init = (
                    len(intersection) / len(dims_final) if len(dims_final) > 0 else 0.0
                )

                # % of initial dimensions retained in final
                pct_init_in_final = (
                    len(intersection) / len(dims_init) if len(dims_init) > 0 else 0.0
                )

                records.append({
                    "env": env,
                    "operator": operator,
                    "seed": seed_idx,
                    "effective_rank_init": k_init,
                    "effective_rank_final": k_final,
                    "overlap": overlap,
                    "pct_final_in_init": pct_final_in_init,
                    "pct_init_in_final": pct_init_in_final,
                })

                print(
                    f"[DONE] {env} | {operator} | seed {seed_idx} "
                    f"(k_init={k_init}, k_final={k_final}, "
                    f"Jaccard={overlap:.3f}, "
                    f"final∈init={pct_final_in_init:.3f}, "
                    f"init∈final={pct_init_in_final:.3f})"
                )

            except Exception as e:
                print(f"[WARN] Failed on {env}/{operator}/seed{seed_idx}: {e}")


# -----------------------------
# Aggregation
# -----------------------------
summary = (
    pd.DataFrame(records)
    .groupby(["env", "operator"])
    .agg(
        mean_effective_rank_init=("effective_rank_init", "mean"),
        mean_effective_rank_final=("effective_rank_final", "mean"),
        mean_jaccard_overlap=("overlap", "mean"),
        mean_pct_final_in_init=("pct_final_in_init", "mean"),
        mean_pct_init_in_final=("pct_init_in_final", "mean"),
        n=("seed", "count"),
    )
    .reset_index()
    .sort_values(["env", "operator"])
)


# -----------------------------
# Output
# -----------------------------
pd.set_option("display.float_format", "{:.3f}".format)

print(f"\n=== Initial vs Final Effective Rank and Overlap (threshold={VARIANCE_THRESHOLD}) ===")
print(summary.to_string(index=False))