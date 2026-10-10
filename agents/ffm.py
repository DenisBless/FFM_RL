"""Fixed-point Flow Matching (FFM) for offline-to-online RL with flow policies.

This file implements the method of "Reinforcement Learning via Fixed-point Flow
Matching" exactly as used for the paper's experiments. It trains a critic Q(s, a)
and a flow policy whose velocity field v_theta(x_t, t | s) is fit by a
flow-matching-style regression against a critic-derived target.

The idea in one paragraph
-------------------------
For a Boltzmann target pi+ ∝ p_ref(a) exp(tau_q Q(s, a)), the velocity field that
generates the noised target is, up to a rescaling of the noisy action, a conditional
average of the target's score at clean actions (the *target flow identity*). That
score is known: it is the reference policy's score plus tau_q * grad_a Q. Flow
matching would regress onto it using samples from pi+, which we do not have. FFM
uses samples from the *current* policy instead and iterates: generate clean actions
with the current flow, label them with the critic gradient, noise them along the
interpolant x_t = t*a_1 + (1-t)*eps, and regress the velocity field onto the label.
The optimal policy is a fixed point of this loop. No backpropagation through the
integrator, no importance weights, no conversion to a diffusion model.

What is in this file
--------------------
* ``critic_loss``   TD(n) critic with a pessimistic ensemble backup (shared with
                    Q-chunking / TRQAM style codebases).
* ``actor_loss``    (a) a BC flow-matching loss that maintains the reference policy
                    ``actor_slow`` on the data, and (b) the FFM regression for the
                    fine-tuned policy ``actor_fast``, written in score space:

        R = [t (v(x_t) - a_1) + (t - c) eps] / (1 - t + delta)
            - c / (t + delta) * s_target(a_1),          loss = mean ||R||^2

    with ``s_target = s_ref(a_1) + tau_q * grad_a Q(s, a_1)`` (reference score plus
    critic gradient, so the fixed point is the product tilt p_ref * exp(tau_q Q)),
    and the variance-optimal mixing schedule

        c(t) = t^2 / (t^2 + (1-t)^2 * I),

    where I is a Fisher-information normaliser: either the fixed constant
    ``i_fisher`` or an EMA of ||s_target||^2 / d (``adaptive_fisher``). ``delta``
    (``t_eps``) softens the endpoint denominators.
* a path-space KL trust region to the reference policy, ``lambda * path_kl``, whose
  multiplier follows projected dual descent on a KL budget (``kl_budget``).

Two regimes are used in the paper and selected through the config:
    locomotion     tau_q=30, fixed Fisher (adaptive_fisher=False, i_fisher=1),
                   trust region on (lam_scale=1), no action chunking
    manipulation   tau_q=1, adaptive Fisher, trust region off offline
                   (lam_scale=0), action chunking (horizon 5)
See scripts/domains.sh for the exact per-domain flags.
"""

import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class FFMAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    lam: float      # trust-region multiplier lambda (dual variable of the KL constraint)
    kl_ema: float   # EMA-smoothed path-space KL estimate
    i_ema: float    # EMA of ||s_target||^2 / d, the Fisher normaliser used by c(t)
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------ critic
    def critic_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=rng)
        next_actions = jnp.clip(next_actions, -1, 1)
        next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], next_actions)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)

        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(batch['observations'], batch_actions, params=grad_params)
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    # ------------------------------------------------------------------- actor
    def get_lambda(self):
        """Trust-region multiplier applied to the KL penalty (lam_scale=0 turns it off)."""
        return self.config["lam_scale"] * self.lam

    def actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng, a_rng, tq_rng, eps_rng = jax.random.split(rng, 6)

        ## BC flow-matching loss: keeps the reference policy (actor_slow) on the data.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_slow')(batch['observations'], x_t, t, params=grad_params)
        flow_loss = jnp.mean(jnp.square(pred - vel).mean(axis=-1) * batch["valid"][..., -1])

        if self.config.get("bc_only", False):
            return flow_loss, {'flow_loss': flow_loss, "fast_loss": 0.0}

        obs = batch["observations"]

        ## (1) Clean actions from the current fine-tuned policy (forward only, no gradients).
        noises = jax.random.normal(a_rng, (batch_size, action_dim))
        a_1 = self.compute_flow_actions(obs, noises, model="fast")
        a_1 = jnp.clip(a_1, -1, 1)

        ## (2) Critic action gradient at the clean actions, from the target critic,
        ##     with the action clipped to the valid range inside the differentiated map.
        grad_fn = jax.grad(
            lambda o, a: self.network.select("target_critic")(o, jnp.clip(a, -1., 1.)).mean(axis=0).sum(), 1)
        qgrad = self.config["tau_q"] * grad_fn(obs, a_1)

        ## (3) Reference score, so the fixed point is the product tilt p_ref * exp(tau_q Q).
        ##     s_ref is read off the (target) reference velocity through the bridge at
        ##     t = 1 - ref_t_delta:  s_ref(a) ~= (v_ref(a, t) - a/t) * t / (1 - t).
        actor_ref = self.network.select("target_actor_slow")
        t_ref = 1.0 - self.config["ref_t_delta"]
        t_ref_arr = jnp.full((batch_size, 1), t_ref)
        v_ref = actor_ref(obs, a_1, t_ref_arr)
        s_ref = (v_ref - a_1 / t_ref) * t_ref / (1.0 - t_ref)
        tscore = qgrad + s_ref

        ## (4) Fisher normaliser for the mixing schedule c(t): second moment of the
        ##     target score per coordinate. The schedule uses the slow EMA from previous
        ##     steps (like a target statistic), or the fixed constant i_fisher.
        i_batch = jnp.mean(jnp.sum(tscore ** 2, axis=-1)) / action_dim
        if self.config["adaptive_fisher"]:
            i_hat = jnp.maximum(self.i_ema, 1e-8)
        else:
            i_hat = self.config["i_fisher"]

        ## (5) Fresh interpolant coupling: num_t_samples (t, eps) pairs per observation.
        rep = self.config["num_t_samples"]
        obs_r = jnp.tile(obs, (rep,) + (1,) * (obs.ndim - 1))
        a_1r = jnp.tile(a_1, (rep, 1))
        tscore_r = jnp.tile(tscore, (rep, 1))
        t_f = jax.random.uniform(tq_rng, (batch_size * rep, 1))
        eps = jax.random.normal(eps_rng, (batch_size * rep, action_dim))
        x_tf = (1 - t_f) * eps + t_f * a_1r

        ## (6) Variance-optimal mixing schedule c*(t) = t^2 / (t^2 + (1-t)^2 I).
        c = t_f ** 2 / (t_f ** 2 + (1 - t_f) ** 2 * i_hat)

        ## (7) FFM residual in score space; delta softens the endpoint denominators.
        delta = self.config["t_eps"]
        v_fine = self.network.select('actor_fast')(obs_r, x_tf, t_f, params=grad_params)
        residual = (t_f * (v_fine - a_1r) + (t_f - c) * eps) / (1 - t_f + delta) \
            - (c / (t_f + delta)) * tscore_r
        ffm_loss = jnp.mean(jnp.sum(jnp.square(residual), axis=-1))

        ## (8) Path-space KL to the reference policy at the same interpolant points.
        ##     2/g(t)^2 = t/(1-t), so E_t[(2/g^2) ||dv||^2] is the path-KL density.
        v_base = actor_ref(obs_r, x_tf, t_f)
        vel_diff_sq = jnp.sum(jnp.square(v_fine - v_base), axis=-1)
        kl_weight = (t_f / (1 - t_f + delta))[..., 0]
        path_kl = jnp.mean(kl_weight * vel_diff_sq)
        horizon = self.config['horizon_length'] if self.config["action_chunking"] else 1
        path_kl = path_kl / horizon

        fast_loss = ffm_loss + self.get_lambda() * path_kl

        info = {
            "ffm_loss": ffm_loss,
            "path_kl": path_kl,
            "i_batch": i_batch,
            "c_mean": c.mean(),
            "qgrad_mean": jnp.abs(qgrad).mean(),
            "qgrad_max": jnp.abs(qgrad).max(),
            "s_ref_mean": jnp.abs(s_ref).mean(),
        }
        return flow_loss + fast_loss, {'flow_loss': flow_loss, "fast_loss": fast_loss, **info}

    # ----------------------------------------------------------------- updates
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        if self.config.get("bc_only", False):
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
            return actor_loss, info

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        return critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    def dual_update(self, kl_estimate):
        """Projected dual descent on the trust-region multiplier lambda."""
        kl_clip_max = self.config["kl_clip_coef"] * self.config["kl_budget"]
        kl_estimate_clipped = jnp.minimum(kl_estimate, kl_clip_max)

        new_kl_ema = (1 - self.config["kl_ema_coef"]) * self.kl_ema + \
                     self.config["kl_ema_coef"] * kl_estimate_clipped

        constraint_violation = new_kl_ema - self.config["kl_budget"]
        new_lam = jnp.maximum(self.config["lambda_min"],
                              self.lam + self.config["eta_lambda"] * constraint_violation)
        new_lam = jnp.minimum(new_lam, self.config["lambda_max"])
        return new_lam, new_kl_ema

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)

        if agent.config.get("bc_only", False):
            return agent.replace(network=new_network, rng=new_rng), info

        agent.target_update(new_network, 'critic')
        agent.target_update(new_network, 'actor_slow')

        kl_estimate = info['actor/path_kl']
        new_lam, new_kl_ema = agent.dual_update(kl_estimate)

        new_i_ema = (1 - agent.config["i_ema_coef"]) * agent.i_ema + \
                    agent.config["i_ema_coef"] * info['actor/i_batch']

        kl_clip_max = agent.config["kl_clip_coef"] * agent.config["kl_budget"]
        info['dual/lambda'] = agent.config["lam_scale"] * new_lam
        info['dual/kl_estimate'] = kl_estimate
        info['dual/kl_clipped'] = (kl_estimate > kl_clip_max).astype(jnp.float32)
        info['dual/kl_ema'] = new_kl_ema
        info['dual/kl_budget'] = agent.config["kl_budget"]
        info['dual/i_ema'] = new_i_ema

        return agent.replace(
            network=new_network, rng=new_rng,
            lam=new_lam, kl_ema=new_kl_ema, i_ema=new_i_ema,
        ), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    # ---------------------------------------------------------------- sampling
    @jax.jit
    def sample_actions(self, observations, rng):
        action_dim = self.config['action_dim'] * \
            (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        noises = jax.random.normal(
            rng,
            (*observations.shape[: -len(self.config['ob_dims'])], self.config["best_of_n"], action_dim),
        )
        observations = jnp.repeat(observations[..., None, :], self.config["best_of_n"], axis=-2)

        actions = self.compute_flow_actions(observations, noises, model="fast")
        actions = jnp.clip(actions, -1, 1)

        # best-of-n by Q (n = 1 in the paper, which makes this a plain sample)
        q = self.network.select("critic")(observations, actions).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)
        bshape = indices.shape
        indices = indices.reshape(-1)
        actions = jnp.reshape(actions, (-1, self.config["best_of_n"], action_dim))[
            jnp.arange(len(indices)), indices, :].reshape(bshape + (action_dim,))
        return actions

    @partial(jax.jit, static_argnames="model")
    def compute_flow_actions(self, observations, noises, model="slow"):
        """Integrate the flow ODE with the Euler scheme from the prior sample `noises`."""
        actions = noises
        networks = [self.network.select(f'actor_{m}') for m in model.split(",")]
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = sum(network(observations, actions, t) for network in networks)
            actions = actions + vels / self.config['flow_steps']
        return jnp.clip(actions, -1, 1)

    # ------------------------------------------------------------------ create
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=config['num_qs'],
        )
        actor_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            layer_norm=config['actor_layer_norm'],
            action_dim=full_action_dim,
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor_fast=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            target_actor_fast=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            actor_slow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            target_actor_slow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)

        if config["clip_grad"]:
            network_tx = optax.chain(optax.clip_by_global_norm(max_norm=1.0),
                                     optax.adam(learning_rate=config["lr"]))
        else:
            network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor_slow'] = params['modules_actor_slow']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(
            rng,
            network=network,
            lam=float(config["lam_init"]),
            kl_ema=0.0,
            i_ema=1.0,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='ffm',
            ob_dims=ml_collections.config_dict.placeholder(list),    # set automatically
            action_dim=ml_collections.config_dict.placeholder(int),  # set automatically

            ## Networks and optimisation
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            value_hidden_dims=(512, 512, 512, 512),
            value_layer_norm=True,
            clip_grad=True,          # clip the global gradient norm at 1

            ## Action chunking (Q-chunking); horizon_length is set from the --horizon_length flag
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=False,

            ## Critic
            num_qs=10,               # critic ensemble size
            rho=0.5,                 # pessimistic backup: mean - rho * std over the ensemble
            discount=0.995,
            tau=0.005,               # target network update rate
            flow_steps=10,           # Euler steps for the flow ODE
            best_of_n=1,             # best-of-n action sampling (1 = plain sampling)

            ## FFM regression
            tau_q=1.0,               # critic scale (inverse temperature) in the target score
            t_eps=0.05,              # endpoint softening of the 1/t and 1/(1-t) denominators
            ref_t_delta=0.1,         # bridge offset at which the reference score is read off
            adaptive_fisher=False,   # True: EMA Fisher normaliser in c(t); False: constant i_fisher
            i_fisher=1.0,            # fixed Fisher normaliser, c(t) = t^2 / (t^2 + (1-t)^2 i_fisher)
            i_ema_coef=0.01,         # EMA coefficient of the adaptive Fisher normaliser
            num_t_samples=4,         # fresh (t, eps) interpolant samples per observation

            ## Path-space KL trust region (projected dual descent on lambda)
            lam_scale=1.0,           # multiplies lambda; 0 disables the trust region
            lam_init=1.0,
            kl_budget=1.0,           # KL budget eps_KL (--kl_budget_online overrides it online)
            eta_lambda=0.01,         # dual step size
            kl_ema_coef=0.1,         # EMA coefficient of the KL estimate
            lambda_min=0.01,
            lambda_max=100.0,
            kl_clip_coef=2.0,        # the KL estimate is clipped at kl_clip_coef * kl_budget
        )
    )
    return config
