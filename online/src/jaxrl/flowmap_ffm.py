"""
Flow-map FFM: the FFM trainer of ffm_submission.py with a flow-MAP policy,
sampled in map_steps = 2 (or 1) network evaluations instead of the 16-step
velocity ODE.

Single-file trainer; config: config/flowmap_ffm.yaml. Critic, TD(lambda)
soft backups, HL-Gauss distributional critic, Q-score clipping, trust-region
dual, noise schedule, squashing and evaluation protocol are FFM's
(ffm_submission.py). The file is the cleaned instantiation of the research
trainer: every branch the configuration never executes and every
diagnostic-only computation has been removed, nothing on the live training
path has been changed, and every jax.random.split of the original is kept
in place. At matched seed on the DMC tasks, with XLA's GPU autotuning off,
its losses and updates match the original up to float rounding: bit for bit
for the two-step sampler, and to about 1e-6 relative in the first updates
for the one-step sampler (the MJX humanoid physics is not reproducible
run to run even then). With autotuning on (the default), not even two runs
of the same file agree bit for bit, and full runs agree statistically.

Time convention as in FFM: X_t = alpha(t) X_1 + beta(t) eps, prior at
t = 0, action at t = 1. The flow map u(x, o, t, r) is the average velocity
of the flow from x at time t to time r >= t,

    X_r = X_t + (r - t) u(X_t, o, t, r),      u(x, o, t, t) = v(x, o, t),

one network for the map and (at r = t) the velocity
(networks/flow_models/flowmap_models.py). What changes against FFM:

1. SAMPLER. Actions a = f(X_1) from K = map_steps Euler steps of the map on
   the grid t_k = k/K, X_{k+1} = X_k + (1/K) u(X_k, t_k, t_{k+1}),
   X_0 ~ N(0, sigma_0^2 I). The training entropy in the soft reward is the
   EXACT entropy of this discrete sampler,
       H = d/2 log(2 pi e sigma_0^2) + sum_k log|det(I + J_k / K)| + log|det J_f(X_1)|,
   Jacobians by jacfwd, |det| floored at logdet_floor (a folding map would
   send it to -inf).

2. ACTOR LOSS. Each minibatch of rollout self-samples X_1 (with the critic
   score g at X_1, as FFM) is split statically. The first 1 - map_frac
   carries FFM's fixed-point regression on the velocity at r = t, in
   velocity form (alpha/D_soft)^2 ||u(X_t, t, t) - b_cond||^2 with FFM's
   conditional velocity
       b_cond = alpha_dot X_1 - (a(t)/alpha) eps + D_soft/(alpha + delta) (c/alpha) g
   (flowmap_models.ffm_cond_velocity; the same loss as ffm_submission's
   score-space residual). The remaining map_frac samples carry the map
   regression at (t, r) = (min, max) of two uniforms: Lagrangian
   self-distillation (Boffi, Albergo & Vanden-Eijnden, 2025) from the
   network's own velocity,
       ||d/dr [X_t + (r - t) u(X_t, t, r)] - sg(u(X_r, r, r))||^2,
       X_r = sg(X_t + (r - t) u(X_t, t, r)),
   scaled by map_weight. The critic reaches the map only through the
   velocity. The trust region is FFM's velocity-space L2 to the
   iteration-start network, at every (X_t, t, r) of the minibatch.

3. TEMPERATURE DUAL. The dual reads the divergence-form entropy of the
   iteration-start VELOCITY ODE (diff_steps midpoint steps, Hutchinson
   probes) on ent_dual_batch rollout states, not the map's exact entropy:
   the velocity is the field the critic drive sharpens directly, and it runs
   ahead of the map it is distilled into; a dual reading the map lowers the
   temperature until the drive Q / temperature explodes. Anti-windup: above
   temp_max the dual may lower the temperature but not raise it.

EVALUATION: deterministic central action, the map sampler run from X_0 = 0
(training rollouts stay stochastic); one episode per eval env.

resume_every > 0 runs the evaluation chunks from Python and checkpoints the
full train state every resume_every chunks (preemptible clusters); a
restarted run in the same directory resumes from it.
"""

import logging
import os
import pickle
import time
import traceback
from typing import Any, Callable

import hydra
import jax
import numpy as np
from flax import nnx, struct
from flax.struct import PyTreeNode
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax import numpy as jnp
from jax.random import PRNGKey
from omegaconf import DictConfig, OmegaConf

import optax
import wandb
from src.env_utils.jax_wrappers import (
    ClipAction,
    LogWrapper,
    MjxGymnaxWrapper,
    NormalizeVec,
)
from src.jaxrl import utils
from src.networks.jax_models import CategoricalCriticNetwork
from src.networks.flow_models.ffm_models import make_schedule
from src.networks.flow_models.flowmap_models import (
    FlowMapActor,
    FlowMapNetwork,
    ffm_cond_velocity,
    flowmap_lsd_target,
)

logging.basicConfig(level=logging.INFO)


class Transition(struct.PyTreeNode):
    obs: jax.Array
    critic_obs: jax.Array
    action: jax.Array
    action_unsquashed: jax.Array
    squashing_correction_grad: jax.Array
    squashing_correction_val: jax.Array
    entropy_est: jax.Array
    reward: jax.Array
    soft_reward: jax.Array
    next_emb: jax.Array
    value: jax.Array
    done: jax.Array
    truncated: jax.Array
    Q_value: jax.Array
    Q_score: jax.Array
    info: dict[str, jax.Array]


class FlowMapConfig(struct.PyTreeNode):
    lr: float
    gamma: float
    total_time_steps: int
    num_steps: int
    lmbda: float
    num_mini_batches: int
    num_envs: int
    Q_score_max_norm: float
    Q_score_max_percentile: float
    max_grad_norm: float | None
    normalize_env: bool
    ent_start: float
    ent_target_mult: float
    kl_start: float
    kl_bound: float
    num_eval: int = 25
    max_episode_steps: int = 1000
    eval_first_episode_only: bool = True
    num_epochs_actor: int = 4
    num_epochs_critic: int = 4
    critic_hidden_dim: int = 512
    vmin: int = -100
    vmax: int = 100
    num_bins: int = 151
    use_critic_norm: bool = True
    num_critic_encoder_layers: int = 2
    num_critic_head_layers: int = 2
    num_critic_pred_layers: int = 2
    aux_loss_mult: float = 1.0

    # logging settings
    log_pnorm: bool = False
    log_gnorm: bool = False

    # diffusion / FFM / flow-map settings (DictConfig)
    diffusion: Any = None


class SACTrainState(struct.PyTreeNode):
    critic: nnx.TrainState
    actor: nnx.TrainState
    actor_target: nnx.TrainState
    iteration: int
    time_steps: int
    last_env_state: EnvState
    last_obs: jax.Array
    last_critic_obs: jax.Array


def make_eval_fn(
    env: Environment,
    max_episode_steps: int,
    reward_scale: float = 1.0,
    eval_first_episode_only: bool = True,
) -> Callable[[jax.random.PRNGKey, SACTrainState, PyTreeNode | None], dict[str, float]]:
    """Evaluation with the CENTRAL deterministic policy: the map sampler is
    run from x0 = 0 (the prior's center) instead of a random draw, so eval
    episodes carry no exploration noise (the flow-policy analogue of
    evaluating a Gaussian policy with its mean action). Training rollouts
    are untouched."""

    def evaluation_fn(
        key: jax.random.PRNGKey,
        train_state: SACTrainState,
        norm_state: PyTreeNode | None,
    ):
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)

        def map_policy(key: PRNGKey, obs: jax.Array) -> jax.Array:
            x0 = jnp.zeros((obs.shape[0], actor_model.action_dim))
            action, *_ = actor_model.map_sample(key, obs, x0=x0, with_entropy=False)
            return action

        def step_env(carry, _):
            key, env_state, obs = carry
            key, act_key, env_key = jax.random.split(key, 3)
            action = map_policy(act_key, obs)

            step_key = jax.random.split(env_key, env.num_envs)
            obs, _, env_state, reward, done, info = env.step(
                step_key, env_state, action, update_stats=False
            )
            return (key, env_state, obs), info

        key, init_key = jax.random.split(key)
        init_key = jax.random.split(init_key, env.num_envs)
        obs, _, env_state = env.reset(init_key, norm_state)

        key, env_key = jax.random.split(key)
        _, infos = jax.lax.scan(
            f=step_env,
            init=(key, env_state, obs),
            xs=None,
            length=max_episode_steps,
        )

        dones = infos["returned_episode"]
        first_done_indices = jnp.argmax(dones, axis=0)
        # Mask if no env ever finished (where argmax returns 0)
        env_has_done = jnp.any(dones, axis=0)
        num_envs = dones.shape[1]

        # Mean/std of each env's FIRST episode only (eval_first_episode_only):
        # counting every episode that fits in the window over-weights short,
        # usually failed episodes (inspection paradox).
        def first_dones_only_agg(metric, agg_fun):
            first_metric = metric[first_done_indices, jnp.arange(num_envs)]
            return agg_fun(first_metric, where=env_has_done)

        def all_done_agg(metric, agg_fun):
            return agg_fun(metric, where=dones)

        if eval_first_episode_only:
            agg = first_dones_only_agg
        else:
            agg = all_done_agg

        return {
            "episode_return": agg(infos["returned_episode_returns"], jnp.mean) * reward_scale,
            "episode_return_std": agg(infos["returned_episode_returns"], jnp.std),
            "episode_length": agg(infos["returned_episode_lengths"], jnp.mean),
            "episode_length_std": agg(infos["returned_episode_lengths"], jnp.std),
            "episode_success": agg(infos["returned_episode_successes"], jnp.mean),
            "episode_success_std": agg(infos["returned_episode_successes"], jnp.std),
            "num_episodes": agg(infos["returned_episode"], jnp.sum),
        }

    return evaluation_fn


def make_init(
    cfg: FlowMapConfig,
    env: Environment,
    env_params: EnvParams = None,
) -> Callable[[jax.Array], SACTrainState]:
    def init(key: jax.random.PRNGKey) -> SACTrainState:
        key, model_key = jax.random.split(key)
        obs_dim = env.observation_space(env_params)[0].shape[0]
        critic_obs_dim = env.observation_space(env_params)[1].shape[0]
        action_dim = env.action_space(env_params).shape[0]

        # The flow map u(x, o, t, r): FFM's velocity MLP with a (t, r - t)
        # time embedding; u(x, o, t, t) is the velocity.
        sm = cfg.diffusion.score_model
        forward_model: nnx.Module = FlowMapNetwork(
            action_dim=action_dim,
            observation_dim=obs_dim,
            num_layers=sm.num_layers,
            num_hid=sm.num_hid,
            time_mode=sm.time_mode,
            time_mlp_input=sm.time_mlp_input,
            num_time_fourier=sm.num_time_fourier,
            time_fourier_range_min=sm.time_fourier_range_min,
            time_fourier_range_max=sm.time_fourier_range_max,
            num_time_hid=sm.num_time_hid,
            num_time_out=sm.num_time_out,
            action_mode=sm.action_mode,
            action_mlp_input=sm.action_mlp_input,
            num_action_fourier=sm.num_action_fourier,
            action_fourier_range_min=sm.action_fourier_range_min,
            action_fourier_range_max=sm.action_fourier_range_max,
            num_action_hid=sm.num_action_hid,
            num_action_out=sm.num_action_out,
            outer_clip=sm.outer_clip,
            inner_clip=sm.inner_clip,
            weight_init=sm.weight_init,
            bias_init=sm.bias_init,
            layer_norm=sm.layer_norm,
            layer_norm_type=sm.layer_norm_type,
            rngs=nnx.Rngs(model_key),
        )

        ffm_cfg = cfg.diffusion.ffm
        fm_cfg = cfg.diffusion.flowmap
        # Geometric interpolant schedule (paper Eq. 59) in ratio mode: the
        # prior std is pinned at sigma_0 and the noise window derived from r.
        noise_schedule = make_schedule(cfg.diffusion)
        actor_kwargs = dict(
            action_dim=action_dim,
            observation_dim=obs_dim,
            fwd_model=forward_model,
            schedule=noise_schedule,
            diff_steps=cfg.diffusion.diff_steps,
            kl_start=cfg.kl_start,
            ent_start=cfg.ent_start,
            squashing=cfg.diffusion.squashing,
            squashing_scale=cfg.diffusion.get("squashing_scale", 1.0),
            t_softening=ffm_cfg.get("t_softening", 0.05),
            temp_min=ffm_cfg.get("temp_min", 0.0),
            map_steps=int(fm_cfg.get("map_steps", 2)),
            logdet_floor=fm_cfg.get("logdet_floor", None),
        )
        actor_networks = FlowMapActor(**actor_kwargs, rngs=nnx.Rngs(model_key))
        actor_target_networks = FlowMapActor(**actor_kwargs, rngs=nnx.Rngs(model_key))

        # HL-Gauss distributional critic (as FFM).
        critic_networks: nnx.Module = CategoricalCriticNetwork(
            obs_dim=critic_obs_dim,
            action_dim=action_dim,
            hidden_dim=cfg.critic_hidden_dim,
            num_bins=cfg.num_bins,
            vmin=cfg.vmin,
            vmax=cfg.vmax,
            use_norm=cfg.use_critic_norm,
            encoder_layers=cfg.num_critic_encoder_layers,
            use_simplical_embedding=False,
            head_layers=cfg.num_critic_head_layers,
            pred_layers=cfg.num_critic_pred_layers,
            use_skip=False,
            rngs=nnx.Rngs(model_key),
        )
        # Partition: dual variables (temperature, trust-region multiplier) get
        # their own optimizer, everything else standard adam.
        def get_optimizer_labels(params):
            def map_fn(path, _):
                path_names = [str(p.key) if hasattr(p, "key") else str(p) for p in path]
                full_path = ".".join(path_names).lower()
                if "log_temperature" in full_path or "log_lagrangian" in full_path:
                    return "temp_optim"
                return "standard_adam"

            return jax.tree_util.tree_map_with_path(map_fn, params)

        actor_tx = optax.adam(cfg.lr)
        lagrange_tx = optax.adam(cfg.lr)

        actor_optimizer = optax.transforms.partition(
            {"standard_adam": actor_tx, "temp_optim": lagrange_tx},
            get_optimizer_labels,
        )

        critic_optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm) if cfg.max_grad_norm else optax.identity(),
            optax.adam(cfg.lr),
        )

        actor_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(actor_networks),
            params=nnx.state(actor_networks),
            tx=actor_optimizer,
        )
        actor_target_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(actor_target_networks),
            params=nnx.state(actor_target_networks),
            tx=optax.set_to_zero(),
        )
        critic_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(critic_networks),
            params=nnx.state(critic_networks),
            tx=critic_optimizer,
        )
        actor_param_count = utils.count_params(actor_trainstate.params)
        critic_param_count = utils.count_params(critic_trainstate.params)

        print(f"\n{'─' * 48}")
        print("  Trainer        : flow-map FFM")
        print(f"  Sampler        : flow map, {int(fm_cfg.get('map_steps', 2))} step(s)")
        print(f"  Action dim     : {action_dim}")
        print(f"  Observation dim: {obs_dim}")
        print(f"  Actor (total)  : {actor_param_count:>12,}")
        print(f"  Critic         : {critic_param_count:>12,}")
        print(f"  Total          : {actor_param_count + critic_param_count:>12,}")
        print(f"{'─' * 48}\n")

        key, env_key = jax.random.split(key)
        env_key = jax.random.split(env_key, cfg.num_envs)
        obs, critic_obs, env_state = env.reset(key=env_key, params=env_params)

        # randomize initial time step to prevent all envs stepping in tandem
        _env_state = env_state.unwrapped()
        key, randomize_steps_key = jax.random.split(key)
        _env_state.info["steps"] = jax.random.randint(
            randomize_steps_key,
            _env_state.info["steps"].shape,
            0,
            cfg.max_episode_steps,
        ).astype(jnp.float32)
        env_state.set_env_state(_env_state)

        return SACTrainState(
            actor=actor_trainstate,
            actor_target=actor_target_trainstate,
            critic=critic_trainstate,
            iteration=0,
            time_steps=0,
            last_env_state=env_state,
            last_obs=obs,
            last_critic_obs=critic_obs,
        )

    return init


def make_train_fn(
    cfg: FlowMapConfig,
    env: Environment,
    eval_env: Environment,
    env_params: EnvParams = None,
    log_callback: Callable[[SACTrainState, dict[str, jax.Array]], None] | None = None,
    num_seeds: int = 1,
    reward_scale: float = 1.0,
):
    env = LogWrapper(env, cfg.num_envs)
    eval_env = LogWrapper(eval_env, cfg.num_envs)
    env = ClipAction(env)
    eval_env = ClipAction(eval_env)
    env = NormalizeVec(env, enable=cfg.normalize_env)
    eval_env = NormalizeVec(eval_env, enable=cfg.normalize_env)

    eval_fn = make_eval_fn(
        eval_env,
        cfg.max_episode_steps,
        reward_scale=reward_scale,
        eval_first_episode_only=cfg.eval_first_episode_only,
    )
    fm_cfg = cfg.diffusion.flowmap
    map_frac = float(fm_cfg.get("map_frac", 0.5))
    if not 0.0 < map_frac < 1.0:
        raise ValueError(f"flowmap.map_frac must be in (0, 1), got {map_frac}")
    action_dim_total = jnp.prod(jnp.array(env.action_space(env_params).shape))
    # Entropy target of the temperature dual: H* = -ent_target_mult * dim(A).
    action_size_target = action_dim_total * cfg.ent_target_mult
    # Hutchinson probe count of the velocity entropy read by the dual.
    ent_div_probes = int(cfg.diffusion.ffm.get("ent_div_probes", 4))

    def collect_rollout(
        key: PRNGKey, train_state: SACTrainState
    ) -> tuple[Transition, SACTrainState]:
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        # SARSA-style bootstrap values: the critic as of the previous
        # iteration's update (the rollout runs before this iteration's).
        critic_model = nnx.merge(
            train_state.critic.graphdef, train_state.critic.params
        )

        def step_env(carry, _) -> tuple[tuple, Transition]:
            key, env_state, obs, critic_obs, act = carry
            key, next_act_key, ent_key, step_key = jax.random.split(key, 4)
            del ent_key  # unused; kept for RNG parity
            step_key = jax.random.split(step_key, cfg.num_envs)
            # flow-map action (stochastic through the prior draw), sampled at
            # the previous step and carried forward: one sampler call per
            # env step instead of two.
            action, action_unsquashed, squashing_correction_grad, squashing_correction_val, _ = act

            next_obs, next_critic_obs, next_env_state, reward, done, info = env.step(
                step_key, env_state, action
            )

            action = jnp.clip(action, -0.999, 0.999)

            # next action (executed at the next step in the carry-forward path:
            # SARSA-style on-policy bootstrap) with the per-state TRAINING
            # entropy: the exact log-det entropy of the discrete map sampler.
            next_act = actor_model.map_sample(next_act_key, next_obs)
            next_action, _, _, _, entropy_est = next_act

            next_emb, _, _, value = critic_model.forward(next_critic_obs, next_action)

            # soft Bellman: r + gamma * temperature * H(pi(.|o'))
            backup_temp = actor_model.temperature()
            soft_reward = reward + cfg.gamma * entropy_est * backup_temp

            transition = Transition(
                obs=obs,
                critic_obs=critic_obs,
                action=action,
                action_unsquashed=action_unsquashed,
                squashing_correction_grad=squashing_correction_grad,
                squashing_correction_val=squashing_correction_val,
                entropy_est=entropy_est,
                next_emb=next_emb,
                reward=reward,
                soft_reward=soft_reward,
                value=value,
                done=done,
                truncated=next_env_state.truncated,
                Q_value=jnp.zeros_like(value),
                Q_score=jnp.zeros_like(action),
                info=info,
            )
            return (key, next_env_state, next_obs, next_critic_obs, next_act), transition

        key, init_key = jax.random.split(key)
        init_carry_tail = actor_model.map_sample(init_key, train_state.last_obs)

        rollout_state, transitions = jax.lax.scan(
            f=step_env,
            init=(
                key,
                train_state.last_env_state,
                train_state.last_obs,
                train_state.last_critic_obs,
                init_carry_tail,
            ),
            length=cfg.num_steps,
        )
        _, last_env_state, last_obs, last_critic_obs, _ = rollout_state
        train_state = train_state.replace(
            last_env_state=last_env_state,
            last_obs=last_obs,
            last_critic_obs=last_critic_obs,
            time_steps=train_state.time_steps + cfg.num_steps * cfg.num_envs,
        )

        return transitions, train_state

    def learn_step(
        key: PRNGKey, train_state: SACTrainState, batch: Transition
    ) -> tuple[SACTrainState, dict[str, jax.Array]]:
        # TD(lambda) returns on the soft rewards, backwards over the rollout
        # (the N-step soft return of paper Eq. 49; target gradients stopped).
        def compute_nstep_lambda(carry, transition):
            lambda_return, truncated = carry
            done = transition.done
            reward = transition.soft_reward
            value = transition.value
            lambda_sum = cfg.lmbda * lambda_return + (1 - cfg.lmbda) * value
            delta = cfg.gamma * jnp.where(truncated, value, (1.0 - done) * lambda_sum)
            lambda_return = reward + delta
            truncated = transition.truncated
            return (lambda_return, truncated), lambda_return

        _, target_values = jax.lax.scan(
            compute_nstep_lambda,
            (batch.value[-1], jnp.ones_like(batch.truncated[0])),
            batch,
            reverse=True,
        )
        # Reshape data to (num_steps * num_envs, ...)
        data = (batch, target_values)
        data = jax.tree.map(
            lambda x: x.reshape((cfg.num_steps * cfg.num_envs, *x.shape[2:])), data
        )

        # The iteration-start actor: rollout policy, trust-region anchor, and the
        # sampler whose terminal law feeds the fixed-point regression.
        train_state = train_state.replace(
            actor_target=train_state.actor_target.replace(params=train_state.actor.params),
        )
        actor_target_model = nnx.merge(
            train_state.actor_target.graphdef, train_state.actor_target.params
        )

        # Entropy read by the temperature dual: divergence-form entropy of the
        # iteration-start velocity ODE u(x, t, t) (diff_steps midpoint steps,
        # Hutchinson probes) on a subsample of the rollout states.
        key, sub_key, div_key = jax.random.split(key, 3)
        n_sub = int(fm_cfg.get("ent_dual_batch", 256))
        idx_sub = jax.random.randint(sub_key, (n_sub,), 0, data[0].obs.shape[0])
        h_vel = jax.lax.stop_gradient(
            actor_target_model.entropy_divergence(
                div_key, data[0].obs[idx_sub], probes=ent_div_probes, exact=False
            ).mean()
        )

        def update_critic(train_state, key) -> tuple[SACTrainState, dict[str, jax.Array]]:
            def minibatch_update(carry, indices):
                idx, train_state = carry
                minibatch, target_values = jax.tree.map(
                    lambda x: jnp.take(x, indices, axis=0), data
                )

                def critic_loss_fn(params):
                    critic_model = nnx.merge(train_state.critic.graphdef, params)
                    critic_pred = critic_model.critic_cat(
                        minibatch.critic_obs, minibatch.action
                    ).squeeze()
                    # HL-Gauss distributional critic: cross-entropy to the
                    # binned TD(lambda) targets.
                    target_cat = jax.vmap(
                        utils.hl_gauss, in_axes=(0, None, None, None)
                    )(target_values, cfg.num_bins, cfg.vmin, cfg.vmax)
                    critic_update_loss = optax.softmax_cross_entropy(
                        critic_pred, target_cat
                    )

                    # Aux loss: next-embedding prediction (REPPO)
                    _, pred, pred_rew, value = critic_model.forward(
                        minibatch.critic_obs, minibatch.action
                    )
                    aux_loss = optax.squared_error(pred, minibatch.next_emb)
                    aux_loss = jnp.mean(
                        (1 - minibatch.done.reshape(-1, 1)) * aux_loss, axis=-1
                    )

                    critic_loss = optax.squared_error(value, target_values)
                    critic_loss = jnp.mean(critic_loss)
                    loss = jnp.mean(
                        (1.0 - minibatch.truncated)
                        * (critic_update_loss + cfg.aux_loss_mult * aux_loss)
                    )
                    metrics = dict(
                        value_loss=critic_loss,
                        critic_update_loss=critic_update_loss,
                        loss=loss,
                        aux_loss=aux_loss,
                        q=value.mean(),
                        reward_mean=minibatch.reward.mean(),
                        target_values=target_values.mean(),
                        q_min=value.min(),
                        q_max=value.max(),
                        target_values_min=target_values.min(),
                        target_values_max=target_values.max(),
                        reward_min=minibatch.reward.min(),
                        reward_max=minibatch.reward.max(),
                    )

                    if cfg.log_pnorm:
                        metrics["critic_pnorm"] = utils.tree_norm(params)

                    return loss, metrics

                critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
                output, critic_grads = critic_grad_fn(train_state.critic.params)
                critic_train_state = train_state.critic.apply_gradients(critic_grads)
                train_state = train_state.replace(critic=critic_train_state)
                critic_metrics = output[1]
                if cfg.log_gnorm:
                    critic_metrics["critic_gnorm"] = utils.tree_norm(critic_grads)

                return (idx + 1, train_state), {**critic_metrics}

            key, shuffle_key = jax.random.split(key)
            mini_batch_size = (cfg.num_steps * cfg.num_envs) // cfg.num_mini_batches
            indices = jax.random.permutation(shuffle_key, cfg.num_steps * cfg.num_envs)
            minibatch_idxs = jax.tree.map(
                lambda x: x.reshape((cfg.num_mini_batches, mini_batch_size, *x.shape[1:])),
                indices,
            )

            train_state, metrics = jax.lax.scan(minibatch_update, train_state, minibatch_idxs)
            metrics = jax.tree.map(lambda x: x.mean(0), metrics)
            return train_state, metrics

        def update_actor(train_state, key) -> tuple[SACTrainState, dict[str, jax.Array]]:
            def minibatch_update(carry, indices_and_key):
                indices, key = indices_and_key
                idx, train_state = carry
                minibatch, target_values = jax.tree.map(
                    lambda x: jnp.take(x, indices, axis=0), data
                )

                def flowmap_step(params):
                    # ---- 1. Merge models ----
                    actor_model = nnx.merge(train_state.actor.graphdef, params)
                    old_model = actor_target_model
                    ffm_cfg = cfg.diffusion.ffm
                    sched = actor_model.schedule
                    action_dim = actor_model.action_dim

                    batch_size = minibatch.action.shape[0]

                    # X_1: clean latent self-samples from the rollout policy
                    # (= the iteration-start actor), with the critic gradient
                    # and squash-Jacobian score at X_1.
                    x_1 = minibatch.action_unsquashed
                    obs = minibatch.obs
                    Q_score = minibatch.Q_score
                    squash_grad = minibatch.squashing_correction_grad
                    Q_value = minibatch.Q_value

                    key_t, key_eps, key_ent_x, key_ent_q, key_kl = jax.random.split(key, 5)
                    del key_ent_x, key_ent_q, key_kl  # unused; kept for RNG parity

                    temperature = actor_model.temperature()
                    temp_sg = jax.lax.stop_gradient(temperature)

                    # ---- 2. Terminal target score in latent space ----
                    # g = grad_x [Q(o, f(x))/temp + log|det J_f(x)|]; the Jacobian
                    # term is the product-target reference score (analytic).
                    tscore = Q_score / temp_sg + squash_grad

                    # Fixed Fisher scale I_bar = i_fisher and endpoint
                    # softening delta (as FFM).
                    i_hat = ffm_cfg.get("i_fisher", 1.0)
                    delta = ffm_cfg.get("t_softening", 0.05)

                    # ---- 3. Fresh interpolant samples, (t, r) pairs ----
                    # Static split of the minibatch: the first n_vel samples
                    # carry FFM's velocity regression at r = t, the remaining
                    # n_map the flow-map regression at (t, r) = (min, max) of
                    # two uniforms.
                    n_map = int(round(map_frac * batch_size))
                    n_vel = batch_size - n_map
                    key_t, key_r, key_pin = jax.random.split(key_t, 3)
                    del key_pin  # unused; kept for RNG parity
                    t_a = jax.random.uniform(key_t, (batch_size, 1))
                    t_b = jax.random.uniform(key_r, (batch_size, 1))
                    is_map = (jnp.arange(batch_size) >= n_vel)[:, None]
                    t = jnp.where(is_map, jnp.minimum(t_a, t_b), t_a)
                    r = jnp.where(is_map, jnp.maximum(t_a, t_b), t)
                    eps = jax.random.normal(key_eps, (batch_size, action_dim))
                    alpha_t = sched.alpha(t)
                    beta_t = sched.beta(t)
                    x_t = alpha_t * x_1 + beta_t * eps
                    # FFM's variance-optimal mixing (logging) and softened D
                    c = alpha_t**2 / (alpha_t**2 + beta_t**2 * i_hat)
                    D_soft = sched.D_soft(t, delta)

                    # ---- 4. FFM conditional velocity (velocity-space target) ----
                    # b_cond = alpha_dot X_1 - (a/alpha) eps + D_soft/(alpha+delta) (c/alpha) g;
                    # FFM's score-space residual is (alpha/D_soft)(v - b_cond).
                    b_cond = ffm_cond_velocity(sched, t, x_1, eps, tscore, delta, i_hat)

                    # ---- 5a. Velocity samples: FFM's fixed-point regression ----
                    xv, ov, tv = x_t[:n_vel], obs[:n_vel], t[:n_vel]
                    v_pred = jax.vmap(actor_model.fwd_model, in_axes=(0, 0, 0))(xv, ov, tv)
                    v_old = jax.vmap(old_model.fwd_model, in_axes=(0, 0, 0))(xv, ov, tv)
                    residual = (alpha_t[:n_vel] / D_soft[:n_vel]) * (v_pred - b_cond[:n_vel])
                    unscaled_ffm_loss = jnp.sum(jnp.square(residual), axis=-1)
                    ffm_loss = jnp.mean(unscaled_ffm_loss)

                    # ---- 5b. Map samples: Lagrangian self-distillation ----
                    # The map's endpoint X = x_t + (r - t) u(x_t, t, r) follows
                    # the network's own velocity, d/dr X = sg(u(X, r, r)): one
                    # JVP of u in r, a noise-free target.
                    xm, om, tm, rm = x_t[n_vel:], obs[n_vel:], t[n_vel:], r[n_vel:]
                    u_pred, dX_dr, v_end = flowmap_lsd_target(actor_model.fwd_model, xm, om, tm, rm)
                    map_err = jnp.sum(jnp.square(dX_dr - v_end), axis=-1)
                    u_old = jax.vmap(old_model.fwd_model, in_axes=(0, 0, 0, 0))(xm, om, tm, rm)
                    map_loss = jnp.mean(map_err)
                    map_weight = float(fm_cfg.get("map_weight", 1.0))
                    fit_loss = ffm_loss + map_weight * map_loss

                    # ---- 6. Trust region ----
                    # Velocity-space L2 to the iteration-start network at the
                    # same (X_t, t, r), velocity and map samples alike
                    # (uniform weight, factor 1/2).
                    pred_all = jnp.concatenate([v_pred, u_pred], axis=0)
                    old_all = jnp.concatenate([v_old, u_old], axis=0)
                    kl_surrogate = jnp.mean(
                        0.5 * jnp.sum(jnp.square(pred_all - old_all), axis=-1)
                    )
                    kl_loss = kl_surrogate

                    # Dual-descent multiplier (alternating descent on theta,
                    # ascent on nu).
                    lagrangian = actor_model.lagrangian()
                    kl_scale = jax.lax.stop_gradient(lagrangian).squeeze()

                    actor_loss = fit_loss + kl_scale * kl_loss

                    # ---- 7. Entropy and KL duals ----
                    # Temperature: descend temp * (H - H*) with the velocity
                    # ODE's divergence entropy H = h_vel (stop-gradiented).
                    # temp_max (anti-windup): Adam moves log-temp at a constant
                    # rate whatever the deficit, so a collapse would wind the
                    # temperature up by decades, and it would take as long to
                    # unwind as the collapse lasted. Above temp_max the dual
                    # may lower it, not raise it.
                    target_entropy = h_vel - h_star
                    temp_max = fm_cfg.get("temp_max", None)
                    if temp_max is not None:
                        target_entropy = jnp.where(
                            (temp_sg >= temp_max) & (target_entropy < 0.0), 0.0, target_entropy)
                    target_entropy_loss = temperature * jax.lax.stop_gradient(target_entropy)
                    target_entropy_loss = target_entropy_loss.mean()

                    lagrangian_loss = -lagrangian * jax.lax.stop_gradient(
                        kl_loss - cfg.kl_bound
                    )
                    lagrangian_loss = lagrangian_loss.mean()

                    loss = actor_loss + target_entropy_loss + lagrangian_loss

                    # ---- 8. Logging ----
                    drive_term = (c / (alpha_t + delta)) * tscore
                    metrics = dict(
                        actor_loss=actor_loss,
                        loss=loss,
                        ffm_loss=ffm_loss,
                        map_loss=map_loss,
                        map_err_max=jnp.max(map_err),
                        kl=kl_loss,
                        kl_loss=kl_loss,
                        scaled_kl_loss=kl_scale * kl_loss,
                        loss_ratio=ffm_loss / (kl_scale * kl_loss + 1e-8),
                        m_step_lagrangian=lagrangian,
                        m_step_lagrangian_loss=lagrangian_loss,
                        temp=temperature,
                        # the dual's entropy (velocity ODE) and the rollout
                        # sampler's exact entropy (flow map)
                        entropy=h_vel,
                        entropy_rollout=minibatch.entropy_est.mean(),
                        entropy_rollout_min=jnp.min(minibatch.entropy_est),
                        entropy_loss=target_entropy_loss,
                        target_entropy=h_star,
                        i_hat=i_hat,
                        c_mean=c.mean(),
                        tau_eff=1.0 / temp_sg,
                        nabla_Q_norm=jnp.mean(jnp.sum(jnp.square(minibatch.Q_score), axis=-1)),
                        tscore_norm=jnp.mean(jnp.sum(jnp.square(tscore), axis=-1)),
                        squashing_correction_grad_norm=jnp.mean(
                            jnp.sum(jnp.square(minibatch.squashing_correction_grad), axis=-1)
                        ),
                        mean_Q_vals=jnp.mean(Q_value),
                        std_Q_vals=jnp.std(Q_value),
                        max_Q_val=jnp.max(Q_value),
                        reward_mean=minibatch.reward.mean(),
                        soft_reward_mean=minibatch.soft_reward.mean(),
                        abs_batch_action=jnp.abs(minibatch.action).mean(),
                        abs_batch_action_unsquashed=jnp.abs(minibatch.action_unsquashed).mean(),
                        ctrl_norm=0.5 * jnp.mean(jnp.sum(jnp.square(pred_all), axis=-1)),
                        old_ctrl_norm=0.5 * jnp.mean(jnp.sum(jnp.square(old_all), axis=-1)),
                        max_unscaled_ffm_loss=jnp.max(unscaled_ffm_loss),
                        max_scaled_nabla_Q_normsq=jnp.max(
                            jnp.sum(jnp.square(minibatch.Q_score / temp_sg), axis=-1)
                        ),
                        max_kl_loss=jnp.max(0.5 * jnp.sum(jnp.square(pred_all - old_all), axis=-1)),
                        max_unsquashed_action=jnp.max(jnp.abs(minibatch.action_unsquashed)),
                        drive_term_norm=jnp.mean(
                            jnp.sum(jnp.square(drive_term), axis=-1)
                        ),
                    )

                    if cfg.log_pnorm:
                        metrics["actor_pnorm"] = utils.tree_norm(params)

                    return loss, metrics

                actor_grad_fn = jax.value_and_grad(flowmap_step, has_aux=True)
                output, actor_grads = actor_grad_fn(train_state.actor.params)
                actor_train_state = train_state.actor.apply_gradients(actor_grads)
                actor_metrics = output[1]
                train_state = train_state.replace(actor=actor_train_state)
                if cfg.log_gnorm:
                    actor_metrics["actor_gnorm"] = utils.tree_norm(actor_grads)
                return (idx + 1, train_state), {**actor_metrics}

            key, shuffle_key = jax.random.split(key)
            mini_batch_size = (cfg.num_steps * cfg.num_envs) // cfg.num_mini_batches
            indices = jax.random.permutation(shuffle_key, cfg.num_steps * cfg.num_envs)
            minibatch_idxs = jax.tree.map(
                lambda x: x.reshape((cfg.num_mini_batches, mini_batch_size, *x.shape[1:])),
                indices,
            )

            train_state, metrics = jax.lax.scan(
                minibatch_update,
                train_state,
                (minibatch_idxs, jax.random.split(key, cfg.num_mini_batches)),
            )
            metrics = jax.tree.map(lambda x: x.mean(0), metrics)
            return train_state, metrics

        def update_Q(train_state: SACTrainState, data: Transition, key: PRNGKey):
            """Compute Q values and clipped Q-scores at the stored clean latents.

            Q_score = grad_x Q(o, f(x)) at x = action_unsquashed (through the
            squashing, but without the log|det J| term, which is stored
            separately as squashing_correction_grad).
            """
            del key  # unused (kept in the caller's split for RNG parity)
            critic_model = nnx.merge(
                train_state.critic.graphdef, train_state.critic.params
            )
            actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)

            def Q_score_clipping(grads, max_norm=1.0, percentile=95.0):
                sample_norms = jnp.linalg.norm(grads, axis=-1, keepdims=True)
                batch_percentile = jnp.percentile(sample_norms, percentile)
                clip_threshold = jnp.minimum(batch_percentile, max_norm)
                scale_factor = jnp.where(
                    sample_norms > clip_threshold,
                    clip_threshold / (sample_norms + 1e-6),
                    1.0,
                )
                return grads * scale_factor, jnp.mean(scale_factor < 1.0)

            def single_sample_critic(obs, action_unsquashed):
                action = actor_model.squashing.forward(action_unsquashed)
                q_val = critic_model.critic(obs, action)
                return jnp.squeeze(q_val)

            dq_da_fn = jax.value_and_grad(single_sample_critic, argnums=1)
            batched_dq_da_fn = jax.vmap(dq_da_fn, in_axes=(0, 0))

            Q_value, Q_score = batched_dq_da_fn(
                data[0].critic_obs, data[0].action_unsquashed
            )
            Q_value, Q_score = (
                jax.lax.stop_gradient(Q_value),
                jax.lax.stop_gradient(Q_score),
            )
            Q_score_clipped, pct_clipped = Q_score_clipping(
                Q_score, cfg.Q_score_max_norm, cfg.Q_score_max_percentile
            )

            return data[0].replace(Q_value=Q_value, Q_score=Q_score_clipped)

        # critic epochs
        key, train_key = jax.random.split(key)
        (_, train_state), update_metrics_critic = jax.lax.scan(
            f=update_critic,
            init=(1, train_state),
            xs=jax.random.split(train_key, cfg.num_epochs_critic),
        )

        # Q values and scores at the stored clean latents (once per iteration)
        key, q_key = jax.random.split(key)
        new_data = update_Q(train_state, data, q_key)
        data = (new_data, data[1])

        # Entropy target of the temperature dual.
        h_star = -action_size_target

        # actor epochs
        key, train_key = jax.random.split(key)
        (_, train_state), update_metrics_actor = jax.lax.scan(
            f=update_actor,
            init=(1, train_state),
            xs=jax.random.split(train_key, cfg.num_epochs_actor),
        )

        update_metrics = {**update_metrics_actor, **update_metrics_critic}
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)

        return train_state, update_metrics

    def train_parts(cfg: FlowMapConfig):
        """(init, loop_body, num_iterations): init(key) -> (train_state, keys),
        loop_body(train_state, key) -> (train_state, metrics) runs one
        evaluation chunk. train_fn scans loop_body over keys in one jit; the
        resumable driver in run() calls it chunk by chunk."""

        def train_eval_step(key, train_state):
            def train_step(
                state: SACTrainState, key: PRNGKey
            ) -> tuple[SACTrainState, dict[str, jax.Array]]:
                key, rollout_key, learn_key = jax.random.split(key, 3)
                transitions, state = collect_rollout(key=rollout_key, train_state=state)
                state, update_metrics = learn_step(
                    key=learn_key, train_state=state, batch=transitions
                )
                metrics = {**update_metrics}
                state = state.replace(iteration=state.iteration + 1)
                return state, metrics

            train_key, eval_key = jax.random.split(key)
            eval_interval = int(
                (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
            )
            eval_time_steps = train_state.time_steps
            if cfg.normalize_env:
                norm_state = train_state.last_env_state
            else:
                norm_state = None

            eval_key, _ = jax.random.split(eval_key)  # second key unused; RNG parity
            eval_metrics = eval_fn(eval_key, train_state, norm_state)

            train_state, train_metrics = jax.lax.scan(
                f=train_step,
                init=train_state,
                xs=jax.random.split(train_key, eval_interval),
            )
            train_metrics = jax.tree.map(lambda x: x[-1], train_metrics)

            train_returns = {
                "train/episode_return": train_state.last_env_state.info[
                    "returned_episode_returns"
                ].mean(),
                "train/episode_length": train_state.last_env_state.info[
                    "returned_episode_lengths"
                ].mean(),
            }

            metrics = {
                "time_step": train_state.time_steps,
                "eval/time_step": eval_time_steps,
                **utils.prefix_dict("train", train_metrics),
                **utils.prefix_dict("eval", eval_metrics),
                **train_returns,
            }
            return train_state, metrics

        def loop_body(
            train_state: SACTrainState, key: PRNGKey
        ) -> tuple[SACTrainState, dict]:
            key, subkey = jax.random.split(key)
            train_state, metrics = jax.vmap(train_eval_step)(
                jax.random.split(subkey, num_seeds), train_state
            )

            jax.debug.callback(log_callback, train_state, metrics)
            return train_state, metrics

        eval_interval = int(
            (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
        )
        num_train_steps = cfg.total_time_steps // (cfg.num_steps * cfg.num_envs)
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )

        def init(key):
            key, init_key = jax.random.split(key)
            train_state = jax.vmap(make_init(cfg, env, env_params))(
                jax.random.split(init_key, num_seeds)
            )
            return train_state, jax.random.split(key, num_iterations)

        return init, loop_body, num_iterations

    def train_fn(key: PRNGKey, cfg: FlowMapConfig) -> tuple[SACTrainState, dict]:
        init, loop_body, _ = train_parts(cfg)
        train_state, keys = init(key)
        state, metrics = jax.lax.scan(f=loop_body, init=train_state, xs=keys)
        return state, metrics

    train_fn.parts = train_parts
    return train_fn


RESUME_FILE = "resume_state.pkl"


def _run_resumable(train_fn, key, hp: FlowMapConfig, every: int):
    """Chunk-by-chunk driver of train_fn (resume_every > 0): the same init and
    per-evaluation loop body, jitted separately and called from Python, with
    the full train state and the metrics so far written atomically to
    RESUME_FILE in the run dir every `every` chunks. A restarted run (e.g.
    after a preemption) resumes from the last checkpoint; the file is removed
    at the end. Same RNG stream as the single-jit scan."""
    init, loop_body, n_iter = train_fn.parts(hp)
    state, keys = jax.jit(init)(key)
    chunk = jax.jit(loop_body, donate_argnums=0)
    start, history = 0, []
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE, "rb") as f:
            ckpt = pickle.load(f)
        leaves, treedef = jax.tree.flatten(state)
        restored = []
        for ref, saved in zip(leaves, ckpt["leaves"], strict=True):
            if jax.dtypes.issubdtype(ref.dtype, jax.dtypes.prng_key):
                restored.append(jax.random.wrap_key_data(saved, impl=jax.random.key_impl(ref)))
            else:
                restored.append(jnp.asarray(saved, dtype=ref.dtype))
        state = jax.tree.unflatten(treedef, restored)
        start, history = ckpt["next"], ckpt["metrics"]
        logging.info(f"Resumed from {RESUME_FILE} at chunk {start}/{n_iter}.")
    for i in range(start, n_iter):
        state, m = chunk(state, keys[i])
        history.append(jax.device_get(m))
        if (i + 1) % every == 0 and i + 1 < n_iter:
            leaves = [
                np.asarray(jax.random.key_data(x)) if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
                else np.asarray(x)
                for x in jax.tree.leaves(state)
            ]
            with open(RESUME_FILE + ".tmp", "wb") as f:
                pickle.dump({"next": i + 1, "leaves": leaves, "metrics": history}, f)
            os.replace(RESUME_FILE + ".tmp", RESUME_FILE)
            logging.info(f"Checkpoint after chunk {i + 1}/{n_iter}.")
    metrics = jax.tree.map(lambda *xs: np.stack(xs), *history)
    if os.path.exists(RESUME_FILE):
        os.remove(RESUME_FILE)
    return state, metrics


def run(cfg: DictConfig) -> float:
    metric_history = []

    def log_callback(state, metrics):
        metrics["sys_time"] = time.perf_counter()
        if len(metric_history) > 0:
            num_env_steps = state.time_steps[0] - metric_history[-1]["time_step"][0]
            seconds = metrics["sys_time"] - metric_history[-1]["sys_time"]
            sps = num_env_steps / seconds
        else:
            sps = 0

        metric_history.append(metrics)
        episode_return = metrics["eval/episode_return"].mean()
        eval_length = metrics["eval/episode_length"].mean()
        eval_success = metrics["eval/episode_success"].mean()

        log_msg = (
            f"step={state.time_steps[0]} "
            f"episode_return={episode_return:.3f}, episode_length={eval_length:.3f}, "
            f"episode_success={eval_success:.3f}"
        )
        log_msg += f" sps={sps:.2f}"
        logging.info(log_msg)

        log_data = {
            "eval/episode_return": episode_return,
            "eval/episode_length": eval_length,
            "sps": sps,
            **jax.tree.map(jnp.mean, utils.filter_prefix("train", metrics)),
        }
        for key, value in metrics.items():
            if key.startswith("eval/"):
                log_data[key] = value.mean() if hasattr(value, "mean") else value

        wandb.log(log_data, step=state.time_steps[0])

    # Set up the experiment: MuJoCo Playground (MJX) environments only (the
    # 20 DMC tasks and the 4 humanoid joystick tasks).
    if cfg.env.type != "mjx":
        raise ValueError(
            f"flowmap_ffm supports env.type 'mjx' only (got {cfg.env.type!r})."
        )
    env = MjxGymnaxWrapper(
        cfg.env.name,
        episode_length=cfg.env.max_episode_steps,
        reward_scale=cfg.env.reward_scaling,
        push_distractions=cfg.env.get("push_distractions", False),
        asymmetric_observation=cfg.env.get("asymmetric_observation", False),
    )
    eval_env = MjxGymnaxWrapper(
        cfg.env.name,
        episode_length=cfg.env.max_episode_steps,
        reward_scale=cfg.env.reward_scaling,
        push_distractions=cfg.env.get("push_distractions", False),
        asymmetric_observation=cfg.env.get("asymmetric_observation", False),
    )

    train_fn = make_train_fn(
        cfg=FlowMapConfig(**cfg.hyperparameters),
        env=env,
        eval_env=eval_env,
        log_callback=log_callback,
        num_seeds=cfg.num_seeds,
        reward_scale=1.0 / cfg.env.reward_scaling,
    )

    wandb.init(
        mode=cfg.wandb.mode,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        tags=[cfg.name, cfg.env.name, cfg.env.type, *cfg.tags],
        config=OmegaConf.to_container(cfg),
        name=f"{cfg.name}-{cfg.env.name.lower()}",
        save_code=True,
    )

    logging.info(OmegaConf.to_yaml(cfg))

    key = jax.random.PRNGKey(cfg.seed)
    start = time.perf_counter()
    resume_every = int(cfg.get("resume_every", 0))
    if resume_every > 0:
        state, metrics = _run_resumable(
            train_fn, key, FlowMapConfig(**cfg.hyperparameters), resume_every
        )
    else:
        state, metrics = jax.jit(train_fn, static_argnums=(1,))(
            key, FlowMapConfig(**cfg.hyperparameters)
        )
        jax.block_until_ready(metrics)
    duration = time.perf_counter() - start

    logging.info(f"Training took {duration:.2f} seconds.")
    jnp.savez("metrics.npz", **metrics)
    wandb.finish()

    return metrics["eval/episode_return"][:, -1].mean().item()


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="flowmap_ffm",
)
def main(cfg: DictConfig):
    try:
        run(cfg)
    except Exception as ex:
        traceback.print_tb(ex.__traceback__)
        traceback.print_exception(ex)


if __name__ == "__main__":
    main()
