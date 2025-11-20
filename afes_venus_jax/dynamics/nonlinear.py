"""Nonlinear advection tendencies evaluated in grid space."""
from __future__ import annotations
from functools import partial
import jax
import jax.numpy as jnp
from .. import spectral, grid, config

_MAX_WIND_SPEED = 400.0  # m/s cap to keep the toy model stable
_MAX_GRADIENT = 1e-3  # limit on scalar gradients to avoid CFL blow-ups


def _upwind_gradient(field, velocity, axis, spacing, periodic):
    """Return a first-order upwind derivative along the requested axis."""

    forward = jnp.roll(field, -1, axis=axis)
    backward = jnp.roll(field, 1, axis=axis)

    if not periodic:
        idx_last = [slice(None)] * field.ndim
        idx_last[axis] = -1
        idx_first = [slice(None)] * field.ndim
        idx_first[axis] = 0
        forward = forward.at[tuple(idx_last)].set(field[tuple(idx_last)])
        backward = backward.at[tuple(idx_first)].set(field[tuple(idx_first)])

    diff_pos = (field - backward) / spacing
    diff_neg = (forward - field) / spacing
    grad = jnp.where(velocity >= 0, diff_pos, diff_neg)
    return jnp.clip(grad, -_MAX_GRADIENT, _MAX_GRADIENT)


def _advect_scalar(q, u, v, lats, lons):
    dlat, dlon = lats[1] - lats[0], lons[1] - lons[0]
    dq_dlon = _upwind_gradient(q, u, axis=-1, spacing=dlon, periodic=True)
    dq_dlat = _upwind_gradient(q, v, axis=-2, spacing=dlat, periodic=False)
    cosphi = grid.cosine_latitudes(lats)[:, None]
    a = config.planet.radius
    return -(u * dq_dlon / (a * cosphi) + v * dq_dlat / a)


@partial(jax.jit, static_argnums=1)
def nonlinear_tendencies(state, cfg: config.ModelConfig | None = None):
    """Compute nonlinear tendencies in grid space then project to spectral."""
    if cfg is None:
        cfg = config.DEFAULT
    lats, lons, _ = grid.gaussian_grid()
    psi, chi = spectral.psi_chi_from_vort_div(state.zeta, state.div)
    u, v = spectral.uv_from_psi_chi(psi, chi)
    u = jnp.clip(u, -_MAX_WIND_SPEED, _MAX_WIND_SPEED)
    v = jnp.clip(v, -_MAX_WIND_SPEED, _MAX_WIND_SPEED)

    def level_tend(zeta_l, div_l, T_l, u_l, v_l):
        dzeta = _advect_scalar(zeta_l.real, u_l, v_l, lats, lons)
        ddiv = _advect_scalar(div_l.real, u_l, v_l, lats, lons)
        dT = _advect_scalar(T_l.real, u_l, v_l, lats, lons)
        return dzeta, ddiv, dT

    dzeta, ddiv, dT = jax.vmap(level_tend)(state.zeta, state.div, state.T, u, v)
    if cfg.numerics.dealiasing:
        dzeta = spectral.apply_two_thirds_filter(dzeta)
        ddiv = spectral.apply_two_thirds_filter(ddiv)
        dT = spectral.apply_two_thirds_filter(dT)
    dzeta = spectral.analysis_grid_to_spec(dzeta)
    ddiv = spectral.analysis_grid_to_spec(ddiv)
    dT = spectral.analysis_grid_to_spec(dT)
    dlnps = spectral.analysis_grid_to_spec(jnp.zeros_like(state.lnps.real))
    return dzeta, ddiv, dT, dlnps
