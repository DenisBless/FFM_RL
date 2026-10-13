import distrax
import flax
import jax
import jax.numpy as jnp


def describe(values: jnp.ndarray, axis: tuple | int = 0) -> dict[str, jnp.ndarray]:
    """Compute basic statistics for a batch of values."""
    return {
        "mean": jnp.mean(values, axis=axis),
        "std": jnp.std(values, axis=axis),
        "min": jnp.min(values, axis=axis),
        "max": jnp.max(values, axis=axis),
    }


def merge_dicts(*prefix_dicts: tuple[str, dict], sep: str = "/") -> dict:
    """Merge metric dictionaries with a prefix for each key."""
    return {
        f"{prefix if prefix else ''}{sep if prefix else ''}{key}": value
        for prefix, metrics in prefix_dicts
        for key, value in metrics.items()
    }


def prefix_dict(prefix: str, metrics: dict, sep: str = "/") -> dict:
    """Add a prefix to all keys in a dictionary."""
    return {f"{prefix}{sep}{key}": value for key, value in metrics.items()}


def postfix_dict(postfix: str, metrics: dict, sep: str = "/") -> dict:
    """Add a postfix to all keys in a dictionary."""
    return {f"{key}{sep}{postfix}": value for key, value in metrics.items()}


def filter_prefix(prefix: str, metrics: dict, sep: str = "/") -> dict:
    """Filter keys in a dictionary by a prefix."""
    return {
        key: value for key, value in metrics.items() if key.startswith(prefix + sep)
    }


def hl_gauss(inp, num_bins, vmin, vmax, epsilon=0.0):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    x = jnp.clip(inp, vmin, max=vmax).squeeze() / (1 - epsilon)
    bin_width = (vmax - vmin) / (num_bins - 1)
    sigma_to_final_sigma_ratio = 0.75
    support = jnp.linspace(
        vmin - bin_width / 2, vmax + bin_width / 2, num_bins + 1, dtype=jnp.float32
    )
    sigma = bin_width * sigma_to_final_sigma_ratio
    cdf_evals = jax.scipy.special.erf((support - x) / (jnp.sqrt(2) * sigma))
    z = cdf_evals[-1] - cdf_evals[0]
    target_probs = cdf_evals[1:] - cdf_evals[:-1]
    target_probs = (target_probs / z).reshape(*inp.shape[:-1], num_bins)

    uniform = jnp.ones_like(target_probs) / num_bins

    return (1 - epsilon) * target_probs + epsilon * uniform

def projection(next_dist, rewards, dones, truncated, ent_coef, next_log_prob, gamma, v_min, v_max, num_atoms, support):
    delta_z = (v_max - v_min) / (num_atoms - 1)
    batch_size = rewards.shape[0]

    entr_bon = - (1 - dones[:, None]) * gamma * ent_coef * next_log_prob.reshape(-1,1)

    # Compute target_z
    target_z = jnp.clip(rewards[:,None] + entr_bon + (1 - dones[:, None]) * gamma * support, a_min=v_min, a_max=v_max)
    b = (target_z - v_min) / delta_z
    l = jnp.floor(b).astype(jnp.int32)
    u = jnp.ceil(b).astype(jnp.int32)

    # Adjust l and u to ensure they remain within valid bounds
    l = jnp.where((u > 0) & (l == u), l - 1, l)
    u = jnp.where((l < (num_atoms - 1)) & (l == u), u + 1, u)

    # Create the projected distribution
    proj_dist = jnp.zeros_like(next_dist)

    # Offset calculation for batch indexing
    offset = jnp.arange(batch_size)[:, None] * num_atoms
    # offset = jnp.tile(offset, (1, num_atoms))  # Repeat along the second axis

    # Index updates for proj_dist
    l_idx = (l + offset).ravel()
    u_idx = (u + offset).ravel()

    # Flattened updates
    l_update = (next_dist * (u.astype(jnp.float32) - b)).ravel()
    u_update = (next_dist * (b - l.astype(jnp.float32))).ravel()

    # Flatten proj_dist for updates
    proj_dist_flat = proj_dist.ravel()

    # Add values to proj_dist
    proj_dist_flat = proj_dist_flat.at[l_idx].add(l_update)
    proj_dist_flat = proj_dist_flat.at[u_idx].add(u_update)

    # Reshape back to [batch_size, num_atoms]
    proj_dist = proj_dist_flat.reshape(batch_size, num_atoms)

    return proj_dist


@flax.struct.dataclass
class MultiSampleLogProb:
    policy_action: jax.Array
    policy_action_log_prob: jax.Array
    action: jax.Array


def fast_multi_log_prob(
    key: jax.Array,
    loc: jax.Array,
    scale: jax.Array,
    offset_scale: jax.Array,
) -> MultiSampleLogProb:
    """Computes 3 samples from a tanh squashed function
    - transformed loc and log_prob
    - sample with base scale
    - sample with scaled scale
    Args:
        key: JAX PRNG key.
        loc: Location of the distribution.
        scale: Scale parameter of the distribution.
        offset_scale: Offset scale for the distribution.
    """
    # log det factor

    # sample base gaussian noise with log prob
    base_noise, base_log_prob = distrax.Normal(
        jnp.zeros_like(loc), scale
    ).sample_and_log_prob(seed=key)
    base_log_prob = jnp.sum(base_log_prob, axis=-1)

    # sample with base scale
    base_sample = loc + base_noise
    base_sample_transformed = jnp.tanh(base_sample)
    # numerically stable jax tanh det jacobian https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
    base_log_prob -= jnp.sum(
        2.0 * (jnp.log(2.0) - base_sample - jax.nn.softplus(-2.0 * base_sample)),
        axis=-1,
    )

    return MultiSampleLogProb(
        policy_action=base_sample_transformed,
        policy_action_log_prob=base_log_prob,
        action=jnp.tanh(loc + offset_scale * base_noise),
    )


def multi_softmax(x, dim=8, get_logits=False):
    inp_shape = x.shape
    if dim is not None:
        x = x.reshape(*x.shape[:-1], -1, dim)
    if get_logits:
        x = jax.nn.log_softmax(x, axis=-1)
    else:
        x = jax.nn.softmax(x, axis=-1)
    return x.reshape(*inp_shape)


def multi_log_softmax(x, dim=8):
    if dim is not None:
        x = x.reshape(*x.shape[:-1], -1, dim)
        return jax.nn.log_softmax(x).reshape(x.shape)
    else:
        return jax.nn.log_softmax(x, axis=-1)


def simplical_softmax_cross_entropy(pred, target, dim=8):
    """Computes the cross-entropy loss for simplical softmax."""
    shape = pred.shape[-1]
    if dim is not None:
        pred = pred.reshape(*pred.shape[:-1], -1, dim)
        target = target.reshape(*target.shape[:-1], -1, dim)
    return jnp.sum(-target * jax.nn.log_softmax(pred, axis=-1), axis=-1).mean() / (
        shape / dim
    )

def tree_norm(tree):
    return jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(tree)))

def count_params(params):
    return sum([p.size for p in jax.tree_util.tree_leaves(params)])


def compute_reverse_ess(log_weights):
    # Subtract the maximum log weight for numerical stability
    max_log_weight = jnp.max(log_weights)
    stable_log_weights = log_weights - max_log_weight

    # Compute the importance weights in a numerically stable way
    is_weights = jnp.exp(stable_log_weights)

    # Compute the sums needed for ESS
    sum_is_weights = jnp.sum(is_weights)
    sum_is_weights_squared = jnp.sum(is_weights ** 2)

    # Calculate the effective sample size (ESS)
    ess = (sum_is_weights ** 2) / (log_weights.shape[0] * sum_is_weights_squared)

    return ess


def save_final_checkpoint(state, cfg, filename: str = "checkpoint_final.pkl") -> str:
    """Pickle the final-iteration model of a REPPO/FFM-style trainer.

    state: final train state (PyTreeNode) with .actor/.critic/.critic_target
    nnx.TrainState fields and .last_env_state; cfg: the resolved hydra
    DictConfig. Written to `filename` relative to the current working
    directory (= the hydra run dir under chdir: true); returns the absolute
    path.

    Contents are pure-numpy pytrees, so loading needs neither jax nor the
    model code: actor_params / critic_params / critic_target_params (rebuild
    the modules via the trainer's make_init and swap the params in),
    time_steps, the resolved config, and -- when normalize_env is set -- the
    NormalizeVec running stats (mean/var/critic_mean/critic_var/count): the
    networks act on whitened observations and are unusable without them. All
    arrays keep the leading num_seeds axis.
    """
    import os
    import pickle

    from omegaconf import OmegaConf

    ckpt = {
        "actor_params": jax.device_get(state.actor.params),
        "critic_params": jax.device_get(state.critic.params),
        "critic_target_params": jax.device_get(state.critic_target.params),
        "time_steps": jax.device_get(state.time_steps),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if cfg.hyperparameters.normalize_env:
        ns = state.last_env_state
        ckpt["obs_normalizer"] = jax.device_get(
            dict(
                mean=ns.mean,
                var=ns.var,
                critic_mean=ns.critic_mean,
                critic_var=ns.critic_var,
                count=ns.count,
            )
        )
    with open(filename, "wb") as f:
        pickle.dump(ckpt, f)
    return os.path.join(os.getcwd(), filename)