"""Semi-implicit gravity-wave solver.

The implementation follows the split-explicit/semi-implicit structure
used in AFES-style spectral cores. The fast hydrostatic gravity-wave
terms coupling divergence and surface pressure are treated implicitly
with off-centering ``alpha`` while all other tendencies remain explicit.
This greatly relaxes the gravity-wave CFL constraint compared with the
previous placeholder that merely scaled tendencies by ``(1 - alpha)``.
"""
from __future__ import annotations

import jax.numpy as jnp

import afes_venus_jax.config as cfg
import afes_venus_jax.state as state
import afes_venus_jax.tendencies as tend

def _gravity_wave_linear_terms(mstate: state.ModelState):
    """Return linearised gravity-wave tendencies for ``div``, ``T``, ``lnps``.

    The linear operator retains only the fast hydrostatic coupling between
    divergence and surface pressure using a vertically varying reference
    temperature profile. This mirrors the external-mode treatment in AFES
    while keeping the implementation compact for this demonstration core.
    """

    lmax = mstate.div.shape[-2] - 1
    ell = jnp.arange(lmax + 1)[:, None]
    lap_eigs = ell * (ell + 1) / (cfg.a ** 2)
    T_ref = tend._reference_temperature_profile(mstate.div.shape[0])[:, None, None]

    # Divergence feels surface pressure gradients via geopotential; use a
    # dry hydrostatic approximation with reference temperature.
    div_lin = -(cfg.R_gas * T_ref) * lap_eigs[None, :, :] * mstate.lnps[None, :, :]

    # Compressional heating linearised about the reference temperature.
    kappa = cfg.R_gas / cfg.cp
    T_lin = -kappa * T_ref * mstate.div

    # Surface-pressure tendency from mass continuity (mean divergence).
    lnps_lin = -jnp.mean(mstate.div, axis=0)
    return div_lin, T_lin, lnps_lin


def semi_implicit_step(mstate: state.ModelState, tendencies, alpha: float = cfg.alpha):
    """Apply semi-implicit off-centering of fast gravity-wave modes."""

    zeta_t, div_t, T_t, lnps_t = tendencies
    div_lin, T_lin, lnps_lin = _gravity_wave_linear_terms(mstate)

    # Separate nonlinear (explicit) from linearised (implicit) components.
    div_nonlin = div_t - div_lin
    T_nonlin = T_t - T_lin
    lnps_nonlin = lnps_t - lnps_lin

    dt = cfg.dt
    lmax = mstate.div.shape[-2] - 1
    ell = jnp.arange(lmax + 1)[:, None]
    lap_eigs = ell * (ell + 1) / (cfg.a ** 2)
    T_ref = tend._reference_temperature_profile(mstate.div.shape[0])[:, None, None]
    kappa = cfg.R_gas / cfg.cp

    # Right-hand sides include explicit nonlinear terms and the explicit
    # portion of the linear operator as per the off-centering factor.
    div_rhs = mstate.div + dt * (div_nonlin + (1 - alpha) * div_lin)
    lnps_rhs = mstate.lnps + dt * (lnps_nonlin + (1 - alpha) * lnps_lin)

    # Solve the coupled (div, lnps) system analytically for each spectral
    # mode: div' = div_rhs - α dt C lnps', lnps' = lnps_rhs - α dt <div'>.
    C = (cfg.R_gas * T_ref) * lap_eigs[None, :, :]
    mean_C = jnp.mean(C, axis=0)
    mean_div_rhs = jnp.mean(div_rhs, axis=0)
    denom = 1.0 - (alpha * dt) ** 2 * mean_C
    lnps_new = (lnps_rhs - alpha * dt * mean_div_rhs) / denom
    div_new = div_rhs - alpha * dt * C * lnps_new[None, :, :]

    # Temperature is forced by compressional heating from the updated
    # divergence but keeps other terms explicit.
    T_rhs = mstate.T + dt * (T_nonlin + (1 - alpha) * T_lin)
    T_new = T_rhs - alpha * dt * kappa * T_ref * div_new

    zeta_new = mstate.zeta + dt * zeta_t
    return state.ModelState(zeta_new, div_new, T_new, lnps_new)
