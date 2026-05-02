from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from qdax.types import Genotype, RNGKey


def exponential_crossover_mask(
    key: jax.Array,
    genotype_length: int,
    cross_lambda: float,
) -> jnp.ndarray:
    """
    Generate exponential multi-point crossover mask.
    """

    max_crossovers = genotype_length

    distances = jax.random.exponential(key, shape=(max_crossovers,))

    expected_crossovers = jnp.maximum(
        1,
        jnp.array(cross_lambda * genotype_length).astype(jnp.int32),
    )

    valid = jnp.arange(max_crossovers) < expected_crossovers
    distances = jnp.where(valid, distances, jnp.inf)

    cumulative_positions = jnp.cumsum(distances)

    last = cumulative_positions[expected_crossovers - 1]
    cumulative_positions = cumulative_positions / last * (genotype_length - 1)

    crossover_points = jnp.clip(
        cumulative_positions.astype(jnp.int32),
        0,
        genotype_length - 1,
    )

    positions = jnp.arange(genotype_length)
    crossover_count = jnp.sum(
        positions[:, None] >= crossover_points[None, :],
        axis=1,
    )

    return (crossover_count % 2) == 0


def isolinecross_variation(
    x1: Genotype,
    x2: Genotype,
    random_key: RNGKey,
    iso_sigma: float,
    line_sigma: float,
    cross_lambda: float,
    cross_prob: float,
    minval: Optional[float] = None,
    maxval: Optional[float] = None,
) -> Tuple[Genotype, RNGKey]:
    """
    Iso+Line+Crossover Variation Operator over a set of pairs of genotypes

    Combines isotropic Gaussian mutation with Line mutation
    probabalistic multi-point crossover, randomly selecting between two offspring.

    Parameters:
        x1 (Genotypes): first batch of genotypes
        x2 (Genotypes): second batch of genotypes
        random_key (RNGKey): RNG key for reproducibility
        iso_sigma (float): spread parameter (noise)
        line_sigma (float): line parameter (direction of the new genotype)
        cross_lambda (float): crossover parameter (length of regions to exchange)
        cross_prob (float): probability of crossover mutation
        minval (float, Optional): minimum value to clip the genotypes
        maxval (float, Optional): maximum value to clip the genotypes

    Returns:
        x (Genotypes): new genotypes
        random_key (RNGKey): new RNG key
    """
    # Computing line_noise
    random_key, key_line_noise = jax.random.split(random_key)
    batch_size = jax.tree_util.tree_leaves(x1)[0].shape[0]
    line_noise = jax.random.normal(key_line_noise, shape=(batch_size,)) * line_sigma

    def _variation_fn(
        x1: jnp.ndarray, x2: jnp.ndarray, random_key: RNGKey
    ) -> jnp.ndarray:

        iso_noise = jax.random.normal(random_key, shape=x1.shape) * iso_sigma
        diff = x2 - x1
        x1 = (x1 + iso_noise) + jax.vmap(jnp.multiply)(diff, line_noise)
        x2 = (x2 - iso_noise) + jax.vmap(jnp.multiply)(-diff, line_noise)

        # Crossover
        def crossover(key):
            key_prob, key_mask = jax.random.split(key)

            use_cross = jax.random.uniform(key_prob) < cross_prob

            def cross(_):
                genotype_length = x1.shape[-1]
                mask = exponential_crossover_mask(
                    key_mask, genotype_length, cross_lambda
                )
                return jnp.where(mask, x1, x2)

            return lax.cond(use_cross, cross, lambda _: x1, operand=None)

        x = lax.cond(
            cross_prob > 0.0,
            crossover,
            lambda _: x1,
            random_key,
        )

        # Back in bounds if necessary (floating point issues)
        if (minval is not None) or (maxval is not None):
            x = jnp.clip(x, minval, maxval)

        return x

    # Create tree of random keys
    nb_leaves = len(jax.tree_util.tree_leaves(x1))
    random_key, subkey = jax.random.split(random_key)
    subkeys = jax.random.split(subkey, num=nb_leaves)
    keys_tree = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(x1), subkeys)

    # Apply variation function to each leaf of the tree
    x = jax.tree_util.tree_map(
        lambda y1, y2, key: _variation_fn(y1, y2, key), x1, x2, keys_tree
    )

    return x, random_key
