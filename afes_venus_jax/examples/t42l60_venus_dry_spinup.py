"""Small Venus dry spin-up demo."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from .. import state, config
from ..dynamics import integrators


def main():
    s = state.initial_isothermal()
    # add tiny perturbation
    key = jax.random.PRNGKey(0)
    pert = 1e-6 * (jax.random.normal(key, s.zeta.shape) + 1j * 0.0)
    s = state.ModelState(zeta=s.zeta + pert, div=s.div, T=s.T, lnps=s.lnps)
    final, _ = integrators.integrate(s, nsteps=10, cfg=config.DEFAULT)
    u_max = jnp.max(jnp.abs(final.zeta.real))
    print("max |zeta|", float(u_max))


if __name__ == "__main__":
    main()
