"""
Flow-map policy for FFM: the policy is a flow MAP u(x, o, t, r) sampled in a
few network evaluations instead of a velocity field integrated as an ODE.

Time convention as in ffm_models.py: X_t = alpha(t) X_1 + beta(t) eps with
alpha(0) = 0 (prior at t = 0) and alpha(1) = 1 (action at t = 1). The flow
map is the average velocity of the flow from the state x_t at time t to time
r >= t,

    u(x_t, t, r) = (1 / (r - t)) int_t^r v(x_tau, tau) d tau,   x_r = x_t + (r - t) u,

so one sampler step is x_r = x_t + (r - t) u(x_t, t, r) and a single step
x_1 = x_0 + u(x_0, 0, 1) generates an action. At r = t the map is the
velocity, u(x, t, t) = v(x, t); one network represents both.

The map is learned by Lagrangian self-distillation (Boffi, Albergo &
Vanden-Eijnden, 2025) from the velocity of the same network: the map's
endpoint X(x_t, t, r) = x_t + (r - t) u(x_t, t, r) solves the flow ODE in its
end time, d/dr X = v(X, r), which is regressed with the network's own
velocity at gap 0 as the (stop-gradiented) target,

    || u + (r - t) d_r u - sg(u(X, r, r)) ||^2,

deterministic given x_t. The velocity itself is trained by FFM's fixed-point
regression, so the critic reaches the map only through the velocity.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Literal

from src.networks.flow_models.ffm_models import FFMActor, NoiseSchedule
from src.networks.flow_models.models.jax_control_net import FeatureProjector


class FlowMapNetwork(nnx.Module):
    """The FFM velocity network (ControlNetwork: MLP with Fourier time /
    action features) whose time projector takes TWO inputs, t and the gap
    r - t, each affinely mapped to [-1, 1]: u(x, o, t, r). Called with
    r = None it returns the velocity u(x, o, t, t), so the FFMActor machinery
    (ODE sampler, divergence entropy) works unchanged on it."""

    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        num_layers: int = 2,
        num_hid: int = 64,
        time_mode: Literal["linear", "mlp"] = "mlp",
        time_mlp_input: Literal["linear", "fourier", "both"] = "both",
        num_time_fourier: int = 16,
        time_fourier_range_min: float = 0.1,
        time_fourier_range_max: float = 100.0,
        num_time_hid: int = 32,
        num_time_out: int = 16,
        action_mode: Literal["linear", "mlp"] = "linear",
        action_mlp_input: Literal["linear", "fourier", "both"] = "both",
        num_action_fourier: int = 16,
        action_fourier_range_min: float = 0.1,
        action_fourier_range_max: float = 100.0,
        num_action_hid: int = 32,
        num_action_out: int = 16,
        outer_clip: float = 1e4,
        inner_clip: float = 1e2,
        weight_init: float = 1e-8,
        bias_init: float = 0.0,
        layer_norm: bool = False,
        layer_norm_type: str = "LayerNorm",
        *,
        rngs: nnx.Rngs,
    ):
        self.outer_clip = outer_clip
        # Two time inputs: t and the gap r - t (both in [0, 1]).
        self.time_projector = FeatureProjector(
            input_dim=2,
            mode=time_mode,
            mlp_input_type=time_mlp_input,
            num_fourier_feats=num_time_fourier,
            fourier_range=(time_fourier_range_min, time_fourier_range_max),
            num_hid=num_time_hid,
            num_out=num_time_out,
            rngs=rngs,
        )
        self.action_projector = FeatureProjector(
            input_dim=action_dim,
            mode=action_mode,
            mlp_input_type=action_mlp_input,
            num_fourier_feats=num_action_fourier,
            fourier_range=(action_fourier_range_min, action_fourier_range_max),
            num_hid=num_action_hid,
            num_out=num_action_out,
            rngs=rngs,
        )
        combined_input_dim = (
            self.action_projector.output_dim + observation_dim + self.time_projector.output_dim
        )
        layers = []
        curr_dim = combined_input_dim
        for i in range(num_layers):
            is_last = i == num_layers - 1
            out_dim = action_dim if is_last else num_hid
            layer = nnx.Linear(curr_dim, out_dim, rngs=rngs)
            if is_last:
                layer.kernel.value *= weight_init
                layer.bias.value = jnp.full_like(layer.bias.value, bias_init)
                layers.append(layer)
            else:
                layers.append(layer)
                if layer_norm:
                    layers.append(getattr(nnx, layer_norm_type)(out_dim, rngs=rngs))
                layers.append(nnx.gelu)
            curr_dim = out_dim
        self.state_time_net = nnx.Sequential(*layers)

    def __call__(self, actions, observations, time, r=None):
        """actions (d,), observations (obs_dim,), time () or (1,), r () or (1,)
        or None (= time: the velocity u(x, t, t))."""
        t = jnp.reshape(jnp.asarray(time, dtype=actions.dtype), (-1,))[:1]
        if r is None:
            gap = jnp.zeros_like(t)
        else:
            gap = jnp.reshape(jnp.asarray(r, dtype=actions.dtype), (-1,))[:1] - t
        t_in = jnp.concatenate([2.0 * t - 1.0, 2.0 * gap - 1.0], axis=-1)
        t_emb = self.time_projector(t_in)
        a_emb = self.action_projector(actions)
        x = jnp.concatenate([a_emb, observations, t_emb], axis=-1)
        out = self.state_time_net(x)
        return jnp.clip(out, -self.outer_clip, self.outer_clip)


# ----------------------------------------------------------------------
# Regression targets
# ----------------------------------------------------------------------


def ffm_cond_velocity(sched: NoiseSchedule, t, x_1, eps, g, delta: float, i_hat: float = 1.0):
    """FFM's per-sample target velocity b_cond(X_t; X_1, eps, g): the
    velocity-space form of the fixed-point residual of ffm_submission.py,

        residual = (alpha / D_soft) * (v - b_cond),
        b_cond   = alpha_dot X_1 - (a(t)/alpha) eps
                   + D_soft / (alpha + delta) * (c/alpha) * g,

    with c = alpha^2 / (alpha^2 + beta^2 I), a(t)/alpha = -beta (alpha alpha_dot
    + I beta beta_dot) / (alpha^2 + beta^2 I) and c/alpha = alpha / (alpha^2 +
    beta^2 I), both finite at alpha = 0. g is the latent target score at X_1
    (Q_score / temperature + squash_grad). All inputs (B, 1) / (B, d).
    FFM's loss is the velocity regression  (alpha/D_soft)^2 ||v - b_cond||^2.
    """
    alpha = sched.alpha(t)
    beta = sched.beta(t)
    alpha_dot = sched.alpha_dot(t)
    bbd = sched.beta_beta_dot(t)
    denom = alpha**2 + beta**2 * i_hat
    a_over_alpha = -beta * (alpha * alpha_dot + i_hat * bbd) / denom
    c_over_alpha = alpha / denom
    D_soft = sched.D_soft(t, delta)
    return alpha_dot * x_1 - a_over_alpha * eps + (D_soft / (alpha + delta)) * c_over_alpha * g


def flowmap_lsd_target(fwd_model, x_t, obs, t, r):
    """Lagrangian self-distillation (Boffi, Albergo & Vanden-Eijnden 2025)
    of the flow map from the same network's velocity. With the map's
    endpoint X(x, t, r) = x + (r - t) u(x, t, r), the flow ODE in the end
    time, d/dr X = v(X, r), is regressed with v replaced by the network's own
    velocity at gap 0 (stop-gradiented):

        L = || u + (r - t) d_r u  -  sg( u(X, r, r) ) ||^2.

    One JVP of u(x, t, .) in r. Returns (u, dX_dr, target): the map
    prediction u(x_t, t, r), d/dr X, and the stop-gradiented target; shapes
    (B, d), (B, obs), (B, 1), (B, 1) -> (B, d), (B, d), (B, d).
    """

    def one(x, o, tt, rr):
        g = lambda s: fwd_model(x, o, tt, s)
        u, du_dr = jax.jvp(g, (rr,), (jnp.ones_like(rr),))
        dX = u + (rr - tt) * du_dr
        X_end = jax.lax.stop_gradient(x + (rr - tt) * u)
        v_end = fwd_model(X_end, o, rr)
        return u, dX, v_end

    u, dX, v_end = jax.vmap(one)(x_t, obs, t, r)
    return u, dX, jax.lax.stop_gradient(v_end)


# ----------------------------------------------------------------------
# Actor
# ----------------------------------------------------------------------


class FlowMapActor(FFMActor):
    """FFMActor whose network is a flow map (FlowMapNetwork); adds the
    K-step map sampler with the exact entropy of the implemented discrete
    map. The inherited ODE sampler and divergence entropy act on the
    velocity u(x, t, t)."""

    def __init__(self, *args, map_steps: int = 2, logdet_floor: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.map_steps = int(map_steps)
        # Floor on |det(I + h J_k)| in the exact entropy: a folding map
        # (det -> 0) would otherwise send log|det| -> -inf into the soft
        # reward. None = no floor.
        self.logdet_floor = None if logdet_floor is None else float(logdet_floor)

    def map_sample(self, key, obs: jax.Array, x0: jax.Array = None, steps: int = None,
                   stop_grad: bool = True, with_entropy: bool = True):
        """K-step flow-map sampler on the uniform grid t_k = k/K:

            x_{k+1} = x_k + h u(x_k, t_k, t_k + h),     h = 1/K,

        from x0 ~ N(0, sigma_0^2 I) (or the given x0). Returns (action, x1,
        squash_grad, squash_val, entropy) where entropy is the EXACT
        per-state entropy of the implemented sampler,

            H = d/2 log(2 pi e sigma_0^2) + sum_k log|det(I + h J_k)| + log|det J_f(x1)|,

        J_k the Jacobian of u(., t_k, t_k + h) at x_k (jacfwd: d forward
        passes per step; for K = 1 this is the one-step map's log-det, which
        the divergence/trace form does NOT approximate). with_entropy=False
        skips the Jacobians and returns zeros for the entropy (evaluation).
        All outputs are stop-gradiented when stop_grad.
        """
        K = self.map_steps if steps is None else int(steps)
        h = 1.0 / K
        if x0 is None:
            x0 = self.sigma_0 * jax.random.normal(key, (obs.shape[0], self.action_dim))
        d = self.action_dim

        def squashing_correction(x):
            return self.squashing.forward_log_det_jacobian(x).sum()

        def _one(obs, x0):
            if stop_grad:
                x0 = jax.lax.stop_gradient(x0)

            def step_fn(carry, k):
                x, logdet = carry
                t = k * h
                r = t + h
                f = lambda z: self.fwd_model(z, obs, t, r)
                u = f(x)
                if with_entropy:
                    J = jax.jacfwd(f)(x)
                    _, ld = jnp.linalg.slogdet(jnp.eye(d, dtype=x.dtype) + h * J)
                    if self.logdet_floor is not None:
                        ld = jnp.maximum(ld, jnp.log(self.logdet_floor))
                else:
                    ld = jnp.zeros((), dtype=x.dtype)
                if stop_grad:
                    u = jax.lax.stop_gradient(u)
                    ld = jax.lax.stop_gradient(ld)
                return (x + h * u, logdet + ld), None

            (x1, logdet), _ = jax.lax.scan(
                step_fn, (x0, jnp.zeros((), dtype=x0.dtype)), jnp.arange(K, dtype=jnp.float32)
            )
            action = self.squashing.forward(x1)
            squash_val, squash_grad = jax.value_and_grad(squashing_correction)(x1)
            h_prior = 0.5 * d * jnp.log(2.0 * jnp.pi * jnp.e * self.sigma_0**2)
            entropy = h_prior + logdet + squash_val
            return action, x1, squash_grad, squash_val, entropy

        return jax.vmap(_one, in_axes=(0, 0))(obs, x0)
