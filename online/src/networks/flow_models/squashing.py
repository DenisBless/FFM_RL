"""Squashing bijectors mapping the unconstrained flow output to the
action box (-1, 1)^d.

ErfSI: y = erf(k x), k = 1/(c sigma_0 sqrt(2)). At c = 1 the N(0,
sigma_0^2) prior pushes forward to Uniform(-1, 1) -- the maximum-entropy
default on the box that pairs naturally with the Gaussian interpolant.
TanhSI: the SAC-standard tanh map. IdentitySquashing: no-op (ablations).
"""

import distrax
import jax
from jax import numpy as jnp


class ErfSI(distrax.Bijector):
    """Erf squashing y = erf(k*x) with k = 1/(c * sigma_0 * sqrt(2)).

    c=1 (default): output is Uniform(-1,1) for prior N(0,sigma_0^2) input,
      achieves the minimum Jacobian variance d/2 under the uniformity constraint.
    c>1: output concentrated near 0; Var[C_f] = d/(2*c^4), falling as 1/c^4.
      Use c>1 to trade action-range coverage for lower IS-weight variance.
    See docs/algorithms.tex §7 for the full analysis.
    """

    def __init__(self, sigma_0: float, c: float = 1.0):
        super().__init__(event_ndims_in=0)
        self._sigma_0 = sigma_0
        self._c = float(c)

    @property
    def _k(self):
        return jnp.sqrt(self._sigma_0 ** -2 / 2.0) / self._c

    def forward(self, x):
        return jax.scipy.special.erf(self._k * x)

    def forward_log_det_jacobian(self, x):
        # d/dx erf(k*x) = (2k/sqrt(pi)) * exp(-(kx)^2)
        # log|det J| = log(2k) - 0.5*log(pi) - (kx)^2
        return jnp.log(2.0 * self._k) - 0.5 * jnp.log(jnp.pi) - (self._k * x) ** 2

    def forward_and_log_det(self, x):
        return self.forward(x), self.forward_log_det_jacobian(x)

    def inverse(self, y):
        return jax.scipy.special.erfinv(y) / self._k


class TanhSI(distrax.Bijector):
    """Standard tanh squashing y = tanh(x)."""

    def __init__(self):
        super().__init__(event_ndims_in=0)

    def forward(self, x):
        return jnp.tanh(x)

    def forward_log_det_jacobian(self, x):
        # Numerically stable: 2*(log2 - x - softplus(-2x))
        return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))

    def forward_and_log_det(self, x):
        return self.forward(x), self.forward_log_det_jacobian(x)

    def inverse(self, y):
        return jnp.arctanh(y)


class IdentitySquashing(distrax.Bijector):
    """No-op bijector: y = x, log|J| = 0.  Used for testing SI without squashing."""

    def __init__(self):
        super().__init__(event_ndims_in=0)

    def forward(self, x):
        return x

    def forward_log_det_jacobian(self, x):
        return jnp.zeros_like(x)

    def forward_and_log_det(self, x):
        return x, jnp.zeros_like(x)

    def inverse(self, y):
        return y
