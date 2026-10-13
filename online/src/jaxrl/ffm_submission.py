"""
FFM, as submitted: massively parallel on-policy maximum-entropy RL with a
flow policy trained by fixed-point flow matching.

Single-file trainer at the PUBLISHED configuration of the online
experiments (MuJoCo Playground DMC and MJX humanoid suites). The file is
the cleaned instantiation of the full research trainer: every branch the
published configuration never executes and every diagnostic-only
computation has been removed, nothing on the live training path has been
changed, and every jax.random.split of the original is kept in place, so
a run of this file reproduces the original runs exactly (validated at
matched seed). Config: config/config.yaml.

The iteration (paper Algorithm 1, equation numbers of the Sep-23 draft):

1. ROLLOUT (on-policy, no gradients). Act with x = f(X_1), X_1 ~ pi_theta
   by integrating the sampling ODE (Eq. 5) from X_0 ~ N(0, sigma_0^2 I)
   under the geometric schedule (Eq. 59); f is the erf squash. Per
   transition: the training entropy H(s') by the instantaneous change of
   variables (Eq. 15) along the sampler's own trajectory, divergences by
   HUTCHINSON's estimator (ent_div_probes Rademacher probes per node), a
   SARSA-style bootstrap Q(s', f(x')) by the previous-iteration critic,
   and the soft reward r + gamma * lambda * H(s').

2. TARGETS. TD(lambda) returns on the soft rewards, backwards over the
   rollout (the N-step soft return, Eq. 49).

3. CRITIC. HL-Gauss distributional critic (cross-entropy to the binned
   targets) plus the REPPO auxiliary next-embedding loss, epochs of
   minibatch SGD. A single critic, no target network: the rollout
   bootstrap simply queries it as of the previous iteration's update.

4. Q-SCORES (once, with the updated critic): grad_x Q(o, f(x)) at the
   stored pre-squash rollout actions, norm-clipped at the batch
   percentile.

5. ACTOR (FFM step). The rollout policy is the iteration-start snapshot
   theta_bar (trust-region anchor AND the sampler of the fixed point), so
   the stored actions ARE the X_1 ~ pi_theta_bar samples of Eq. (11).
   With fresh t ~ U(0,1), eps ~ N(0,I), X_t = alpha X_1 + beta eps, the
   loss is the score-field form of the FFM regression (Eq. 61 generalized
   to arbitrary schedules; w(t) = 1/c(t)^2 in velocity units):

       R = [alpha (v_theta(X_t) - alpha_dot X_1) + a(t) eps] / D_soft
           - cstar / (alpha + delta) * g,
       g = grad_x [ Q(o, f(x)) / temperature + log|det J_f(x)| ]  at X_1,

   with the bridge coefficient D = alpha_dot beta^2 - alpha (beta beta_dot)
   = alpha c(t), the endpoint softening D_soft = D + delta sigma_0^2, the
   variance-optimal mixing cstar = alpha^2 / (alpha^2 + beta^2 I_bar) at
   the fixed Fisher scale I_bar = i_fisher = 1 (Eq. 41), and the noise
   coefficient a(t) = -(cstar alpha_dot beta + (1 - cstar) alpha beta_dot),
   written division-free in beta_dot (finite at t = 1). The trust region
   (Eq. 16) is the velocity L2 to theta_bar at the same (X_t, t), under
   the dual-descent multiplier (Eq. 19); the temperature descends on the
   entropy constraint with target H* = -ent_target_mult * action_dim.

6. EVALUATION: deterministic central action, the ODE integrated from
   X_0 = 0 (training rollouts stay stochastic); one episode per eval env.
"""

import logging
import time
import traceback
from typing import Any, Callable

import hydra
import jax
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
from src.networks.flow_models.ffm_models import (
    FFMActor,
    make_schedule,
)
from src.networks.flow_models.models.jax_control_net import ControlNetwork

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


class FFMConfig(struct.PyTreeNode):
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

    # diffusion / FFM settings (DictConfig)
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
    """Evaluation with the CENTRAL deterministic ODE policy: the flow is
    integrated from x0 = 0 (the prior's center) instead of a random draw, so
    eval episodes carry no exploration noise (the flow-policy analogue of
    evaluating a Gaussian policy with its mean action). Training rollouts
    are untouched."""

    def evaluation_fn(
        key: jax.random.PRNGKey,
        train_state: SACTrainState,
        norm_state: PyTreeNode | None,
    ):
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)

        def ode_policy(key: PRNGKey, obs: jax.Array) -> jax.Array:
            x0 = jnp.zeros((obs.shape[0], actor_model.action_dim))
            action, *_ = actor_model.ode_sample(key, obs, stop_grad=True, x0=x0)
            return action

        def step_env(carry, _):
            key, env_state, obs = carry
            key, act_key, env_key = jax.random.split(key, 3)
            action = ode_policy(act_key, obs)

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
    cfg: FFMConfig,
    env: Environment,
    env_params: EnvParams = None,
) -> Callable[[jax.Array], SACTrainState]:
    def init(key: jax.random.PRNGKey) -> SACTrainState:
        key, model_key = jax.random.split(key)
        obs_dim = env.observation_space(env_params)[0].shape[0]
        critic_obs_dim = env.observation_space(env_params)[1].shape[0]
        action_dim = env.action_space(env_params).shape[0]

        # Velocity head: plain MLP with Fourier time/action features.
        forward_model: nnx.Module = ControlNetwork(
            action_dim=action_dim,
            observation_dim=obs_dim,
            num_layers=cfg.diffusion.score_model.num_layers,
            num_hid=cfg.diffusion.score_model.num_hid,
            time_mode=cfg.diffusion.score_model.time_mode,
            time_mlp_input=cfg.diffusion.score_model.time_mlp_input,
            num_time_fourier=cfg.diffusion.score_model.num_time_fourier,
            time_fourier_range_min=cfg.diffusion.score_model.time_fourier_range_min,
            time_fourier_range_max=cfg.diffusion.score_model.time_fourier_range_max,
            num_time_hid=cfg.diffusion.score_model.num_time_hid,
            num_time_out=cfg.diffusion.score_model.num_time_out,
            action_mode=cfg.diffusion.score_model.action_mode,
            action_mlp_input=cfg.diffusion.score_model.action_mlp_input,
            num_action_fourier=cfg.diffusion.score_model.num_action_fourier,
            action_fourier_range_min=cfg.diffusion.score_model.action_fourier_range_min,
            action_fourier_range_max=cfg.diffusion.score_model.action_fourier_range_max,
            num_action_hid=cfg.diffusion.score_model.num_action_hid,
            num_action_out=cfg.diffusion.score_model.num_action_out,
            outer_clip=cfg.diffusion.score_model.outer_clip,
            inner_clip=cfg.diffusion.score_model.inner_clip,
            weight_init=cfg.diffusion.score_model.weight_init,
            bias_init=cfg.diffusion.score_model.bias_init,
            layer_norm=cfg.diffusion.score_model.layer_norm,
            layer_norm_type=cfg.diffusion.score_model.layer_norm_type,
            rngs=nnx.Rngs(model_key),
        )

        ffm_cfg = cfg.diffusion.ffm
        # Geometric interpolant schedule (paper Eq. 59) in ratio mode: the
        # prior std is pinned at sigma_0 and the noise window derived from r.
        noise_schedule = make_schedule(cfg.diffusion)
        actor_networks = FFMActor(
            action_dim=action_dim,
            observation_dim=obs_dim,
            fwd_model=forward_model,
            schedule=noise_schedule,
            diff_steps=cfg.diffusion.diff_steps,
            kl_start=cfg.kl_start,
            ent_start=cfg.ent_start,
            squashing=cfg.diffusion.squashing,
            squashing_scale=cfg.diffusion.get("squashing_scale", 1.0),
            ent_nodes=ffm_cfg.get("ent_nodes", 8),
            ent_t_max=ffm_cfg.get("ent_t_max", 0.95),
            kl_nodes=ffm_cfg.get("kl_nodes", 16),
            kl_t_max=ffm_cfg.get("kl_t_max", 1.0),
            t_softening=ffm_cfg.get("t_softening", 0.05),
            temp_min=ffm_cfg.get("temp_min", 0.0),
            rngs=nnx.Rngs(model_key),
        )
        actor_target_networks = FFMActor(
            action_dim=action_dim,
            observation_dim=obs_dim,
            fwd_model=forward_model,
            schedule=noise_schedule,
            diff_steps=cfg.diffusion.diff_steps,
            kl_start=cfg.kl_start,
            ent_start=cfg.ent_start,
            squashing=cfg.diffusion.squashing,
            squashing_scale=cfg.diffusion.get("squashing_scale", 1.0),
            ent_nodes=ffm_cfg.get("ent_nodes", 8),
            ent_t_max=ffm_cfg.get("ent_t_max", 0.95),
            kl_nodes=ffm_cfg.get("kl_nodes", 16),
            kl_t_max=ffm_cfg.get("kl_t_max", 1.0),
            t_softening=ffm_cfg.get("t_softening", 0.05),
            temp_min=ffm_cfg.get("temp_min", 0.0),
            rngs=nnx.Rngs(model_key),
        )

        # HL-Gauss distributional critic (the published configuration).
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
        fwd_param_count = utils.count_params(nnx.state(forward_model))

        print(f"\n{'─' * 48}")
        print("  Trainer        : FFM (flow policy)")
        print(f"  Action dim     : {action_dim}")
        print(f"  Observation dim: {obs_dim}")
        print(f"  Velocity net   : {fwd_param_count:>12,}")
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
    cfg: FFMConfig,
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
    action_dim_total = jnp.prod(jnp.array(env.action_space(env_params).shape))
    # Entropy target of the temperature dual: H* = -ent_target_mult * dim(A).
    action_size_target = action_dim_total * cfg.ent_target_mult
    # Hutchinson probe count for the training entropy estimator.
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
            step_key = jax.random.split(step_key, cfg.num_envs)
            # flow-policy action (stochastic through the prior draw),
            # sampled at the previous step and carried forward: one
            # ODE rollout per env step instead of two.
            action, action_unsquashed, squashing_correction_grad, squashing_correction_val = act

            next_obs, next_critic_obs, next_env_state, reward, done, info = env.step(
                step_key, env_state, action
            )

            action = jnp.clip(action, -0.999, 0.999)

            # next action (executed at the next step in the carry-forward path:
            # SARSA-style on-policy bootstrap) and the per-state TRAINING
            # entropy: divergence form (paper Eq. 15) along the sampler's own
            # ODE path, Hutchinson probes per midpoint node.
            next_act = actor_model.ode_sample(next_act_key, next_obs, stop_grad=True)
            next_action, next_x1, _, next_squash_val = next_act
            ent_key, div_key = jax.random.split(ent_key)
            entropy_est = actor_model.entropy_divergence(
                div_key, next_obs, probes=ent_div_probes, exact=False
            )

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
        init_carry_tail = actor_model.ode_sample(
            init_key, train_state.last_obs, stop_grad=True
        )

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

                def ffm_step(params):
                    # ---- 1. Merge models ----
                    actor_model = nnx.merge(train_state.actor.graphdef, params)
                    old_model = actor_target_model
                    ffm_cfg = cfg.diffusion.ffm
                    sched = actor_model.schedule
                    sigma_0 = actor_model.sigma_0  # prior std beta(0)
                    action_dim = actor_model.action_dim

                    batch_size = minibatch.action.shape[0]

                    # X_1: clean latent self-samples from the rollout policy
                    # (= pi_theta_bar, the iteration-start actor), with the
                    # critic gradient and squash-Jacobian score at X_1.
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

                    # Fixed Fisher scale I_bar = i_fisher (the published runs
                    # use 1.0: AMDP equal-variance weighting).
                    i_hat = ffm_cfg.get("i_fisher", 1.0)

                    # ---- 3. Fresh interpolant coupling ----
                    t = jax.random.uniform(key_t, (batch_size, 1))
                    eps = jax.random.normal(key_eps, (batch_size, action_dim))
                    # Schedule functions evaluated once at the sampled t; all
                    # (B, 1). bbd = beta * beta_dot stays finite at t = 1 even
                    # when beta_dot alone diverges (geometric schedule).
                    alpha_t = sched.alpha(t)
                    beta_t = sched.beta(t)
                    alpha_dot_t = sched.alpha_dot(t)
                    bbd_t = sched.beta_beta_dot(t)
                    x_t = alpha_t * x_1 + beta_t * eps

                    # ---- 4. Variance-optimal mixing ----
                    # cstar(t) = alpha^2 / (alpha^2 + beta^2 I_bar); linear
                    # schedule: t^2 / (t^2 + (1-t)^2 sigma_0^2 I_bar).
                    c = alpha_t**2 / (alpha_t**2 + beta_t**2 * i_hat)

                    # ---- 5. FFM residual (score space, softened endpoints) ----
                    delta = ffm_cfg.get("t_softening", 0.05)
                    v_pred = jax.vmap(actor_model.fwd_model, in_axes=(0, 0, 0))(x_t, obs, t)
                    v_old = jax.vmap(old_model.fwd_model, in_axes=(0, 0, 0))(x_t, obs, t)

                    # Softened D(t) = alpha c_s(t) + delta sigma_0^2; linear:
                    # sigma_0^2 (1 - t + delta). Residual
                    #   R = [alpha (v - alpha_dot x_1) + a(t) eps] / D_soft
                    #       - c / (alpha + delta) * g
                    # with a(t) = -(c alpha_dot beta + (1-c) alpha beta_dot)
                    # (linear: (t - c) sigma_0).
                    D_soft = sched.D_soft(t, delta)
                    a_eps = -alpha_t * beta_t * (
                        alpha_t * alpha_dot_t + i_hat * bbd_t
                    ) / (alpha_t**2 + beta_t**2 * i_hat)
                    residual = (
                        alpha_t * (v_pred - alpha_dot_t * x_1) + a_eps * eps
                    ) / D_soft - (c / (alpha_t + delta)) * tscore
                    ffm_loss = jnp.sum(jnp.square(residual), axis=-1)
                    unscaled_ffm_loss = ffm_loss

                    ffm_loss = jnp.mean(ffm_loss)

                    # ---- 6. Trust region ----
                    # Velocity-space L2 to the iteration-start actor at the
                    # same interpolant points (uniform t-weight, factor 1/2).
                    kl_surrogate = jnp.mean(
                        0.5 * jnp.sum(jnp.square(v_pred - v_old), axis=-1)
                    )
                    kl_loss = kl_surrogate

                    # Dual-descent multiplier (alternating descent on theta,
                    # ascent on nu).
                    lagrangian = actor_model.lagrangian()
                    kl_scale = jax.lax.stop_gradient(lagrangian).squeeze()

                    actor_loss = ffm_loss + kl_scale * kl_loss

                    # ---- 7. Entropy and KL duals ----
                    # Temperature: descend temp * (H - H*) with the rollout's
                    # divergence-form entropy estimates (stop-gradiented).
                    entropy = minibatch.entropy_est.mean()
                    target_entropy = entropy - h_star
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
                        kl=kl_loss,
                        kl_loss=kl_loss,
                        kl_surrogate=kl_surrogate,
                        scaled_kl_loss=kl_scale * kl_loss,
                        loss_ratio=ffm_loss / (kl_scale * kl_loss + 1e-8),
                        m_step_lagrangian=lagrangian,
                        m_step_lagrangian_loss=lagrangian_loss,
                        temp=temperature,
                        entropy=entropy,
                        entropy_rollout=minibatch.entropy_est.mean(),
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
                        action_norm=jnp.mean(jnp.sum(jnp.square(minibatch.action), axis=-1)),
                        action_norm_unsquashed=jnp.mean(
                            jnp.sum(jnp.square(minibatch.action_unsquashed), axis=-1)
                        ),
                        ctrl_norm=0.5 * jnp.mean(jnp.sum(jnp.square(v_pred), axis=-1)),
                        old_ctrl_norm=0.5 * jnp.mean(jnp.sum(jnp.square(v_old), axis=-1)),
                        max_unscaled_ffm_loss=jnp.max(unscaled_ffm_loss),
                        max_nabla_Q_normsq=jnp.max(
                            jnp.sum(jnp.square(minibatch.Q_score), axis=-1)
                        ),
                        max_scaled_nabla_Q_normsq=jnp.max(
                            jnp.sum(jnp.square(minibatch.Q_score / temp_sg), axis=-1)
                        ),
                        max_kl_loss=jnp.max(0.5 * jnp.sum(jnp.square(v_pred - v_old), axis=-1)),
                        max_ctrl_normsq=jnp.max(jnp.sum(jnp.square(v_pred), axis=-1)),
                        max_unsquashed_action=jnp.max(jnp.abs(minibatch.action_unsquashed)),
                        D_soft_mean=jnp.mean(D_soft),
                        D_soft_min=jnp.min(D_soft),
                        drive_term_norm=jnp.mean(
                            jnp.sum(jnp.square(drive_term), axis=-1)
                        ),
                        max_drive_term_normsq=jnp.max(
                            jnp.sum(jnp.square(drive_term), axis=-1)
                        ),
                    )

                    if cfg.log_pnorm:
                        metrics["actor_pnorm"] = utils.tree_norm(params)

                    return loss, metrics

                actor_grad_fn = jax.value_and_grad(ffm_step, has_aux=True)
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

    def train_fn(key: PRNGKey, cfg: FFMConfig) -> tuple[SACTrainState, dict]:
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
        key, init_key = jax.random.split(key)
        train_state = jax.vmap(make_init(cfg, env, env_params))(
            jax.random.split(init_key, num_seeds)
        )
        keys = jax.random.split(key, num_iterations)
        state, metrics = jax.lax.scan(f=loop_body, init=train_state, xs=keys)
        return state, metrics

    return train_fn


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

    # Set up the experiment: the published runs use MuJoCo Playground (MJX)
    # environments only (the 20 DMC tasks and the 4 humanoid joystick tasks).
    if cfg.env.type != "mjx":
        raise ValueError(
            f"ffm_submission supports env.type 'mjx' only (got {cfg.env.type!r})."
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
        cfg=FFMConfig(**cfg.hyperparameters),
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
    state, metrics = jax.jit(train_fn, static_argnums=(1,))(
        key, FFMConfig(**cfg.hyperparameters)
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
    config_name="config",
)
def main(cfg: DictConfig):
    try:
        run(cfg)
    except Exception as ex:
        traceback.print_tb(ex.__traceback__)
        traceback.print_exception(ex)


if __name__ == "__main__":
    main()
