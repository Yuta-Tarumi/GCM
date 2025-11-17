"""Spectral hyperdiffusion operators."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from . import config


@jax.jit
def hyperdiffusion(flm: jnp.ndarray, tau: float = config.tau_hdiff, order: int = config.order_hdiff):
    """Apply isotropic hyperdiffusion to a spectral field."""
    # approximate using Fourier zonal wavenumbers only
    nlon = flm.shape[-1]
    m = jnp.fft.fftfreq(nlon) * nlon
    k2 = (m / config.a) ** 2
    factor = (k2 ** (order // 2))[None, :]
    max_k = jnp.max(k2)
    nu = 1.0 / (tau * (max_k ** (order // 2) + 1e-12))
    def apply_level(f):
        fhat = jnp.fft.fft(f, axis=-1)
        return jnp.fft.ifft(fhat * (1 - nu * factor), axis=-1)
    return jax.vmap(apply_level)(flm)


@jax.jit
def apply_all(state, tau: float = config.tau_hdiff, order: int = config.order_hdiff):
    """Apply hyperdiffusion to all prognostic variables."""
    return state.__class__(
        zeta=hyperdiffusion(state.zeta, tau, order),
        div=hyperdiffusion(state.div, tau, order),
        T=hyperdiffusion(state.T, tau, order),
        lnps=hyperdiffusion(state.lnps[None, ...], tau, order)[0],
    )
