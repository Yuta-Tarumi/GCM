"""Spectral hyperdiffusion utilities."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from .config import Config


def diffusion_operator(cfg: Config, nlat=None, nlon=None):
    """Precompute spherical-harmonic hyperdiffusion coefficients.

    Extra grid-size arguments are accepted for compatibility with call sites
    that follow the original AFES-Venus API but are ignored here because the
    coefficients depend only on total wavenumber.
    """

    ell = jnp.arange(cfg.Lmax + 1)
    ell_fraction = jnp.sqrt(ell * (ell + 1)) / jnp.sqrt(cfg.Lmax * (cfg.Lmax + 1))
    coeff_ell = -(1.0 / cfg.tau_hdiff) * (ell_fraction ** (2 * cfg.order_hdiff))
    coeff_ell = coeff_ell.at[0].set(0.0)
    return jnp.broadcast_to(coeff_ell[:, None], (cfg.Lmax + 1, cfg.Lmax + 1))


def hyperdiffusion_timescale(cfg: Config, ell_fraction: float) -> float:
    """Return the e-folding time [s] at a given fraction of the max wavenumber.

    The model sets the hyperdiffusion coefficient so that the smallest resolved
    scale (``ell = ellmax``) decays with ``cfg.tau_hdiff``.  At lower wavenumber
    ``ell = ell_fraction * ellmax``, the decay time scales as

    ``tau(ell) = cfg.tau_hdiff * (1 / ell_fraction) ** (2 * cfg.order_hdiff)``.

    Parameters
    ----------
    cfg
        Model configuration.
    ell_fraction
        Fraction of the spectral radius (``0 < ell_fraction <= 1``).

    Returns
    -------
    float
        E-folding time [s] implied by the current hyperdiffusion settings.
    """

    ell_fraction = jnp.clip(jnp.asarray(ell_fraction), 1e-12, 1.0)
    return float(cfg.tau_hdiff * (1.0 / ell_fraction) ** (2 * cfg.order_hdiff))


def apply_diffusion(state, cfg: Config):
    coeff = diffusion_operator(cfg)
    factor = 1.0 + coeff[None, ...] * cfg.dt
    zeta = state.zeta * factor
    div = state.div * factor
    T = state.T * factor
    return state.__class__(zeta=zeta, div=div, T=T, lnps=state.lnps)
