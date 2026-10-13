"""
FFM (fixed-point flow matching) actor for a *general* one-sided Gaussian
interpolant.

The policy is a flow over the latent action space,

    dX_t = v_theta(X_t, o, t) dt,   X_0 ~ N(0, sigma_0^2 I),   a = f(X_1),

with the one-sided interpolant

    X_t = alpha(t) X_1 + beta(t) eps,   alpha(0) = 0, alpha(1) = 1, beta(1) = 0,

where (alpha, beta) come from a NoiseSchedule object instead of the hardcoded
linear pair (t, sigma_0 (1 - t)). The prior std is sigma_0 = beta(0). Score and
velocity are linked by the schedule-dependent bridge

    v(x, t) = (alpha_dot/alpha) x + c_s(t) s(x, t),
    c_s(t)  = (alpha_dot/alpha) beta^2 - beta beta_dot,
    s(x, t) = (alpha v - alpha_dot x) / D(t),      D(t) = alpha c_s(t)
            = (alpha v - alpha_dot x) / (alpha_dot beta^2 - alpha (beta beta_dot)).

All formulas are written in terms of alpha, alpha_dot, beta and the *product*
beta*beta_dot = d(beta^2)/dt / 2, which stays finite at t = 1 even when beta_dot
alone diverges (geometric schedule: beta ~ sqrt(1 - t)). With the linear
schedule these expressions reduce exactly to the original code, e.g.
D = sigma_0^2 (1 - t) and the softened KL weight alpha / (D + delta sigma_0^2)
= t / (sigma_0^2 (1 - t + delta)).

Schedules:
  LinearSchedule(sigma_0):      alpha = t, beta = sigma_0 (1 - t)  (the
                                linear baseline)
  GeometricSchedule(smax, smin): the *bridged* geometric schedule
    (tfm_schedule_notes.pdf, Sec. 2): with r = smax/smin, L2 = 2 log r,
        tau(t)  = smax^2 (1 - exp(-L2 t)),   tau_T = smax^2 - smin^2,
        alpha   = tau/tau_T,   beta = sqrt(tau_T - tau),   sigma_0 = sqrt(tau_T),
    an exact one-sided interpolant that allocates resolution log-uniformly over
    noise levels [smin, smax]. Unlike linear, c_s(1) = smin^2 log r > 0: the
    velocity->score conversion is regular at the action endpoint.

Training does not require log-densities or divergence traces: the actor is
updated by the FFM fixed-point regression (see the trainer), and the entropy
needed for soft Bellman backups follows from the likelihood-free identity

    d/dt H(t) = -E[ v(X_t, t) . s(X_t, t) ]
    H(pi(.|o)) = d/2 log(2 pi e sigma_0^2) - int_0^1 E[v . s] dt + E[log|det J_f|],

estimated by a trapezoidal quadrature over a fixed time grid (the t=0 anchor is
exact because the prior is Gaussian; the score formula above is regular at t=0
for any schedule with alpha(0) = 0).
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
from flax import nnx

from src.networks.flow_models.squashing import (
    ErfSI,
    IdentitySquashing,
    TanhSI,
)


# ----------------------------------------------------------------------
# Noise schedules: the Gaussian channel X_t | X_1 ~ N(alpha X_1, beta^2 I)
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class NoiseSchedule:
    """Interface for one-sided interpolant schedules (alpha, beta).

    Subclasses provide alpha, beta, alpha_dot and beta_beta_dot (= beta *
    d beta/dt, kept as a product because it stays finite where beta_dot alone
    diverges). Everything else is derived:

        D(t)      = alpha_dot beta^2 - alpha (beta beta_dot)   [= alpha c_s(t)]
        D_soft(t) = D(t) + delta sigma_0^2                     [endpoint softening]
        c_s(t)    = D(t) / alpha                                [score->velocity gain]
        sigma_0   = beta(0)                                     [prior std]

    Frozen dataclass so instances hash/compare by value: nnx treats them as
    static graphdef attributes (like the squashing bijector).
    """

    def alpha(self, t):
        raise NotImplementedError

    def beta(self, t):
        raise NotImplementedError

    def alpha_dot(self, t):
        raise NotImplementedError

    def beta_beta_dot(self, t):
        raise NotImplementedError

    def D(self, t):
        """alpha * c_s(t); the (softened) denominator of score-from-velocity."""
        return self.alpha_dot(t) * self.beta(t) ** 2 - self.alpha(t) * self.beta_beta_dot(t)

    def D_soft(self, t, delta):
        """Softened D: reproduces the linear code's t/(sigma_0^2 (1-t+delta))
        as alpha / D_soft."""
        return self.D(t) + delta * self.sigma_0**2

    def c_s(self, t):
        return self.D(t) / self.alpha(t)

    @property
    def sigma_0(self) -> float:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class LinearSchedule(NoiseSchedule):
    """alpha = t, beta = sigma_0 (1 - t): the original linear interpolant."""

    sigma_0_val: float = 1.0

    def alpha(self, t):
        return t

    def beta(self, t):
        return self.sigma_0_val * (1.0 - t)

    def alpha_dot(self, t):
        return jnp.ones_like(t) if hasattr(t, "shape") else 1.0

    def beta_beta_dot(self, t):
        return -(self.sigma_0_val**2) * (1.0 - t)

    @property
    def sigma_0(self) -> float:
        return float(self.sigma_0_val)


@dataclasses.dataclass(frozen=True)
class GeometricSchedule(NoiseSchedule):
    """Bridged geometric schedule (tfm_schedule_notes.pdf, Sec. 2).

    Driftless clock tau(t) = smax^2 (1 - r^{-2t}), pinned at an independent
    Gaussian with sigma_0^2 = tau_T = smax^2 - smin^2:

        alpha(t) = tau(t)/tau_T = (1 - r^{-2t}) / (1 - r^{-2}),
        beta(t)  = sqrt(tau_T - tau(t)) = smax sqrt(r^{-2t} - r^{-2}).

    Derived: alpha_dot = tau_dot/tau_T, beta beta_dot = -tau_dot/2 with
    tau_dot(t) = smax^2 L2 exp(-L2 t) = sigma_t(t)^2 (the VE fresh-noise rate),
    hence D = tau_dot (2 tau_T - tau) / (2 tau_T) and
    c_s = tau_dot (2 tau_T - tau) / (2 tau) > 0 on (0, 1] with
    c_s(1) = smin^2 log r.
    """

    sigma_max: float = 1.0
    sigma_min: float = 0.05

    @property
    def _L2(self) -> float:
        return 2.0 * math.log(self.sigma_max / self.sigma_min)

    @property
    def _tau_T(self) -> float:
        return self.sigma_max**2 - self.sigma_min**2

    def _tau(self, t):
        return self.sigma_max**2 * (1.0 - jnp.exp(-self._L2 * t))

    def _tau_dot(self, t):
        return self.sigma_max**2 * self._L2 * jnp.exp(-self._L2 * t)

    def alpha(self, t):
        return self._tau(t) / self._tau_T

    def beta(self, t):
        # tau(1) = tau_T analytically; clip against float cancellation error.
        return jnp.sqrt(jnp.maximum(self._tau_T - self._tau(t), 0.0))

    def alpha_dot(self, t):
        return self._tau_dot(t) / self._tau_T

    def beta_beta_dot(self, t):
        return -0.5 * self._tau_dot(t)

    @property
    def sigma_0(self) -> float:
        return float(math.sqrt(self._tau_T))


def make_schedule(diffusion_cfg) -> NoiseSchedule:
    """Build a NoiseSchedule from the cfg.diffusion DictConfig.

    Geometric schedule, two parameterizations:
      - ratio mode (preferred for sweeps): `ratio` r > 1 set (non-null).
        The prior std is pinned to `sigma_0` and the noise window derived as
            sigma_max = sigma_0 / sqrt(1 - r^-2),   sigma_min = sigma_max / r,
        so sqrt(sigma_max^2 - sigma_min^2) = sigma_0 exactly. Sweeping `ratio`
        then varies only the noise range, never the latent/prior scale (a
        sigma_max sweep at fixed sigma_min silently rescales sigma_0 = beta(0),
        confounding the two -- observed as the sigma_max=0.5 instability).
      - explicit mode (backward compatible): `ratio` null/absent; `sigma_max`
        and `sigma_min` are used verbatim and sigma_0 = sqrt(smax^2 - smin^2)
        falls out.
    """
    sched_type = diffusion_cfg.get("schedule", "linear")
    if sched_type == "linear":
        return LinearSchedule(sigma_0_val=float(diffusion_cfg.sigma_0))
    elif sched_type == "geometric":
        ratio = diffusion_cfg.get("ratio", None)
        if ratio is not None:
            r = float(ratio)
            if r <= 1.0:
                raise ValueError(
                    f"diffusion.ratio must be > 1 (got {r}); it is sigma_max/sigma_min."
                )
            sigma_0 = float(diffusion_cfg.get("sigma_0", 1.0))
            sigma_max = sigma_0 / math.sqrt(1.0 - r**-2)
            return GeometricSchedule(sigma_max=sigma_max, sigma_min=sigma_max / r)
        return GeometricSchedule(
            sigma_max=float(diffusion_cfg.sigma_max),
            sigma_min=float(diffusion_cfg.sigma_min),
        )
    raise ValueError(
        f"Unknown diffusion.schedule: {sched_type!r}. Choose 'linear' or 'geometric'."
    )


class FFMActor(nnx.Module):
    """Deterministic flow policy trained by fixed-point flow matching,
    generalized to an arbitrary one-sided (alpha, beta) noise schedule."""

    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        fwd_model: nnx.Module,
        schedule: NoiseSchedule,
        diff_steps: int = 16,
        kl_start: float = 0.1,
        ent_start: float = 0.1,
        squashing: str = "erf",
        squashing_scale: float = 1.0,
        ent_nodes: int = 8,
        ent_t_max: float = 0.95,
        kl_nodes: int = 16,
        kl_t_max: float = 1.0,
        t_softening: float = 0.05,
        temp_min: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.fwd_model = fwd_model
        self.diff_steps = diff_steps
        self.dt = 1.0 / diff_steps
        self.schedule = schedule
        # Prior std beta(0); trainer-side uses (OU exploration, entropy anchor,
        # erf squashing scale) read this exactly as before.
        self.sigma_0 = float(schedule.sigma_0)
        self.ent_nodes = ent_nodes
        self.ent_t_max = float(ent_t_max)
        self.kl_nodes = kl_nodes
        self.kl_t_max = float(kl_t_max)
        self.t_softening = float(t_softening)
        self.temp_min = float(temp_min)

        # Dual variables: KL trust-region multiplier and entropy temperature.
        # Same parameter names as SIActor so the partitioned optimizer in the
        # trainer picks them up for the Lagrangian learning rate.
        self.log_lagrangian = nnx.Param(jnp.ones(1) * math.log(kl_start))
        self.log_temperature = nnx.Param(jnp.ones(1) * math.log(ent_start))

        if squashing == "erf":
            self.squashing = ErfSI(self.sigma_0, c=squashing_scale)
        elif squashing == "tanh":
            self.squashing = TanhSI()
        elif squashing == "identity":
            self.squashing = IdentitySquashing()
        else:
            raise ValueError(
                f"Unknown squashing type: {squashing!r}. Choose 'erf', 'tanh', or 'identity'."
            )

    # ------------------------------------------------------------------
    # Interpolant bridge
    # ------------------------------------------------------------------

    def score_from_velocity(self, v, x, t):
        """s = (alpha*v - alpha_dot*x) / (D + 1e-6); regular at t=0 for any
        schedule with alpha(0)=0 (there s = -x/sigma_0^2, the exact prior
        score). Linear: (t*v - x)/(sigma_0^2 (1-t) + 1e-6), singular at t=1;
        geometric: D(1) = smin^2 log r > 0, regular everywhere."""
        sched = self.schedule
        return (sched.alpha(t) * v - sched.alpha_dot(t) * x) / (sched.D(t) + 1e-6)

    # ------------------------------------------------------------------
    # Deterministic ODE sampling (midpoint time grid, Euler steps)
    # ------------------------------------------------------------------

    def ode_sample(
        self,
        key,
        obs: jax.Array,
        stop_grad: bool = True,
        x0: jax.Array = None,
        return_path: bool = False,
    ):
        """Sample actions by integrating the velocity ODE.

        If `x0` is given (batch of latent prior draws, e.g. an OU-correlated
        chain for temporally coherent exploration), it is used instead of a
        fresh N(0, sigma_0^2) draw; per-step marginals are unchanged as long
        as x0 is marginally N(0, sigma_0^2).

        Returns (action, x1_unsquashed, squash_grad, squash_val) with
          squash_val  = log|det J_f(x1)|          (summed over dims)
          squash_grad = grad_x log|det J_f(x1)|   (the product-target reference score)
        With return_path=True, additionally returns path_x: (B, K, d), the
        state at which the velocity was queried at each of the K midpoint
        times t_j = (j + 1/2)/K (input to kl_estimate_path).
        """
        keys = jax.random.split(key, num=obs.shape[0])
        if x0 is None:
            x0 = self.sigma_0 * jax.random.normal(key, (obs.shape[0], self.action_dim))

        def squashing_correction(x):
            return self.squashing.forward_log_det_jacobian(x).sum()

        def _ode_sample(key, obs, x0):
            if stop_grad:
                x0 = jax.lax.stop_gradient(x0)

            def step_fn(x, step):
                t = (step + 0.5) * self.dt
                v = self.fwd_model(x, obs, t)
                if stop_grad:
                    v = jax.lax.stop_gradient(v)
                return x + v * self.dt, x

            x1, xs = jax.lax.scan(
                step_fn, x0, jnp.arange(self.diff_steps, dtype=jnp.float32)
            )
            action = self.squashing.forward(x1)
            squash_val, squash_grad = jax.value_and_grad(squashing_correction)(x1)
            return action, x1, squash_grad, squash_val, xs

        action, x1, squash_grad, squash_val, xs = jax.vmap(
            _ode_sample, in_axes=(0, 0, 0)
        )(keys, obs, x0)
        if return_path:
            return action, x1, squash_grad, squash_val, xs
        return action, x1, squash_grad, squash_val

    # ------------------------------------------------------------------
    # Likelihood-free entropy estimate (trapezoidal quadrature)
    # ------------------------------------------------------------------

    def entropy_estimate(self, key, obs: jax.Array, x1: jax.Array, squash_val: jax.Array):
        """Per-state entropy estimate of the squashed policy.

        obs: (B, obs_dim), x1: (B, d) terminal latents sampled from the policy,
        squash_val: (B,) log|det J_f(x1)|. Fresh interpolation noise per node.
        Everything is evaluated in inference mode; the caller decides whether
        gradients flow (the trainer only uses this inside stop_gradient).
        """
        batch_size, d = x1.shape
        num_nodes = self.ent_nodes
        t_nodes = jnp.linspace(0.0, self.ent_t_max, num_nodes)
        # trapezoid weights on the uniform grid
        h = self.ent_t_max / (num_nodes - 1)
        w = jnp.full((num_nodes,), h).at[0].set(h / 2).at[-1].set(h / 2)

        eps = jax.random.normal(key, (num_nodes, batch_size, d))

        def node(t, eps_k):
            x_t = self.schedule.alpha(t) * x1 + self.schedule.beta(t) * eps_k
            t_arr = jnp.full((batch_size, 1), t)
            v = jax.vmap(self.fwd_model, in_axes=(0, 0, 0))(x_t, obs, t_arr)
            s = self.score_from_velocity(v, x_t, t)
            return jnp.sum(v * s, axis=-1)  # (B,)

        integrand = jax.vmap(node, in_axes=(0, 0))(t_nodes, eps)  # (K, B)

        h_prior = 0.5 * d * jnp.log(2.0 * jnp.pi * jnp.e * self.sigma_0**2)
        h_latent = h_prior - jnp.einsum("k,kb->b", w, integrand)
        return h_latent + squash_val

    # ------------------------------------------------------------------
    # Likelihood-free KL estimate between flow policies (paper Eq. 12)
    # ------------------------------------------------------------------

    def kl_estimate(self, key, obs: jax.Array, x1: jax.Array, old_fwd_model):
        """Per-state KL(pi_theta || pi_old) between two flow policies.

        Paper Eq. (12): D_KL = int_0^1 1/c_s(t) E_{rho_t}[||v - v_old||^2] dt.
        The expectation is under the *current* policy's interpolant marginals:
        x1 are fresh terminal self-samples (B, d), interpolation noise is
        fresh per node.

        The weight is softened with the same delta as the FFM loss,
        1/c_s(t) -> alpha / (D + delta sigma_0^2)  [linear:
        t / (sigma_0^2 (1 - t + delta))], so the grid can include t = 1: the
        successive-policy velocity mismatch concentrates at the terminal end
        (the FFM residual weight peaks there), and truncating the tail makes
        the estimate blind exactly where the drive acts. Own grid (kl_nodes on
        [0, kl_t_max], trapezoid), independent of the entropy grid.
        Gradients flow through self.fwd_model only; the caller passes
        stop-gradiented x1 and the old velocity field.
        """
        batch_size, d = x1.shape
        num_nodes = self.kl_nodes
        delta = self.t_softening
        t_nodes = jnp.linspace(0.0, self.kl_t_max, num_nodes)
        h = self.kl_t_max / (num_nodes - 1)
        w = jnp.full((num_nodes,), h).at[0].set(h / 2).at[-1].set(h / 2)

        eps = jax.random.normal(key, (num_nodes, batch_size, d))

        def node(t, eps_k):
            x_t = self.schedule.alpha(t) * x1 + self.schedule.beta(t) * eps_k
            t_arr = jnp.full((batch_size, 1), t)
            v_new = jax.vmap(self.fwd_model, in_axes=(0, 0, 0))(x_t, obs, t_arr)
            v_old = jax.vmap(old_fwd_model, in_axes=(0, 0, 0))(x_t, obs, t_arr)
            # softened 1/c_s(t)
            w_kl = self.schedule.alpha(t) / self.schedule.D_soft(t, delta)
            return w_kl * jnp.sum(jnp.square(v_new - v_old), axis=-1)  # (B,)

        integrand = jax.vmap(node, in_axes=(0, 0))(t_nodes, eps)  # (K, B)
        return jnp.einsum("k,kb->b", w, integrand)

    def kl_estimate_path(self, obs: jax.Array, path_x: jax.Array, old_fwd_model):
        """Per-state KL(pi_theta || pi_old) along the sampler's own ODE path.

        Same Eq. (12) integral as kl_estimate, but evaluated at the (x, t)
        pairs the K-step sampler actually queried while generating x1
        (path_x from ode_sample(..., return_path=True)), instead of fresh
        interpolant draws. Justification: the probability-flow ODE shares the
        interpolant's time marginals (continuity equation), so each path
        state at query time t_j = (j + 1/2)/K is a rho_{t_j} sample -- up to
        the field's self-consistency error and Euler discretization. In
        exchange, the estimate constrains the divergence of the *implemented*
        samplers: velocity differences at (x, t) the sampler never queries
        cannot change generated actions, so there is no between-node loophole
        to game. Deterministic given the path (no fresh node noise); path
        points are correlated along a trajectory, so per-state estimates
        average over the batch only.

        Riemann sum on the sampler's midpoint grid (max t = 1 - 1/(2K) < 1)
        with the softened weight alpha / (D + delta sigma_0^2). path_x:
        (B, K, d), stop-gradiented by the sampler; gradients flow through
        self.fwd_model only.
        """
        batch_size, num_steps, d = path_x.shape
        delta = self.t_softening
        t_nodes = (jnp.arange(num_steps, dtype=jnp.float32) + 0.5) * self.dt
        xs = jnp.moveaxis(path_x, 1, 0)  # (K, B, d)

        def node(t, x_t):
            t_arr = jnp.full((batch_size, 1), t)
            v_new = jax.vmap(self.fwd_model, in_axes=(0, 0, 0))(x_t, obs, t_arr)
            v_old = jax.vmap(old_fwd_model, in_axes=(0, 0, 0))(x_t, obs, t_arr)
            # softened 1/c_s(t)
            w_kl = self.schedule.alpha(t) / self.schedule.D_soft(t, delta)
            return w_kl * jnp.sum(jnp.square(v_new - v_old), axis=-1)  # (B,)

        integrand = jax.vmap(node, in_axes=(0, 0))(t_nodes, xs)  # (K, B)
        return integrand.sum(axis=0) * self.dt

    # ------------------------------------------------------------------
    # Reference entropy via the divergence form (paper App. D, Prop. 15)
    # ------------------------------------------------------------------

    def entropy_divergence(self, key, obs: jax.Array, probes: int = 4, exact: bool = False, discrete: bool = False):
        """Per-state entropy of the K-step sampler via the divergence form,

            H = d/2 log(2 pi e sigma_0^2) + int_0^1 E[div v(X_t, t)] dt
                + log|det J_f(X_1)|

        (paper App. D, Prop. 15: a second integration by parts turns
        -E[v.s] into E[div v]). Unlike the v.s estimators, this identity
        holds for the *implemented* flow's own pushforward marginals with no
        self-consistency assumption -- the continuity equation is automatic
        for the ODE -- so it is exact up to time discretization and the
        divergence estimate, and serves as the reference against which the
        likelihood-free estimators are validated.

        div v per midpoint node: exact=True uses the Jacobian trace via
        jacfwd (d forward passes per node); otherwise Hutchinson's estimator
        with `probes` Rademacher probes, E_eps[eps^T (dv/dx) eps], via jvp
        (`probes` forward passes per node). discrete=True instead accumulates
        log|det(I + h J)| per step (slogdet of the full Jacobian): the EXACT
        entropy of the implemented discrete K-step Euler sampler, of which
        the divergence integral is the O(h) truncation (G1 checkpoint,
        K=16: continuous-div overestimates the discrete sampler by ~5 nats).
        Output is stop-gradiented: diagnostic only, one value per state.
        """
        keys = jax.random.split(key, num=obs.shape[0] + 1)
        x0 = self.sigma_0 * jax.random.normal(keys[0], (obs.shape[0], self.action_dim))

        def _one(key, obs, x0):
            def step_fn(carry, step):
                x, div_int = carry
                t = (step + 0.5) * self.dt
                f = lambda z: self.fwd_model(z, obs, t)
                v = f(x)
                if discrete:
                    # exact per-step entropy change of the DISCRETE Euler map
                    # x -> x + h v: log|det(I + h J)|. The h tr(J) form below
                    # is its O(h) truncation and overestimates the sampler
                    # entropy when the field contracts strongly.
                    J = jax.jacfwd(f)(x)
                    _, ld = jnp.linalg.slogdet(
                        jnp.eye(self.action_dim, dtype=x.dtype) + self.dt * J
                    )
                    return (x + v * self.dt, div_int + ld), None
                if exact:
                    div = jnp.trace(jax.jacfwd(f)(x))
                else:
                    def one_probe(k):
                        eps = (
                            2.0 * jax.random.bernoulli(k, 0.5, (self.action_dim,))
                            .astype(x.dtype) - 1.0
                        )
                        _, jeps = jax.jvp(f, (x,), (eps,))
                        return jnp.sum(eps * jeps)

                    pk = jax.random.split(jax.random.fold_in(key, step), probes)
                    div = jnp.mean(jax.vmap(one_probe)(pk))
                return (x + v * self.dt, div_int + div * self.dt), None

            (x1, div_int), _ = jax.lax.scan(
                step_fn,
                (x0, jnp.zeros((), dtype=x0.dtype)),
                jnp.arange(self.diff_steps, dtype=jnp.float32),
            )
            squash_val = self.squashing.forward_log_det_jacobian(x1).sum()
            h_prior = 0.5 * self.action_dim * jnp.log(
                2.0 * jnp.pi * jnp.e * self.sigma_0**2
            )
            return h_prior + div_int + squash_val

        return jax.lax.stop_gradient(
            jax.vmap(_one, in_axes=(0, 0, 0))(keys[1:], obs, x0)
        )

    # ------------------------------------------------------------------
    # Dual-variable helpers (same interface as SIActor)
    # ------------------------------------------------------------------

    def temperature(self) -> jax.Array:
        # Additive floor: a flow policy has no minimum stochasticity of its
        # own (the sampler is a deterministic map of the prior draw, unlike a
        # Gaussian policy's min_std or an SDE's injected noise), so without a
        # floor the entropy dual can collapse the
        # temperature to ~0 early -- the tilt becomes greedy w.r.t. a still-flat
        # critic and exploration dies irrecoverably. The additive form keeps
        # the dual differentiable and unchanged above the floor.
        return self.temp_min + jnp.exp(self.log_temperature.value)

    def lagrangian(self) -> jax.Array:
        return jnp.exp(self.log_lagrangian.value)
