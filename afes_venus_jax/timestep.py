"""Single time-step driver."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import os

import afes_venus_jax.config as cfg
import afes_venus_jax.state as state
import afes_venus_jax.tendencies as tend
import afes_venus_jax.implicit as implicit
import afes_venus_jax.diffusion as diffusion

STRICT_SANITY = os.getenv("AFES_VENUS_JAX_STRICT_SANITY", "1") != "0"

def step(mstate: state.ModelState, time_seconds: float = 0.0):
    zeta_t, div_t, T_t, lnps_t = tend.compute_nonlinear_tendencies(mstate, time_seconds=time_seconds)
    new_state = implicit.semi_implicit_step(mstate, (zeta_t, div_t, T_t, lnps_t))
    stepped = diffusion.apply_diffusion(new_state)
    _runtime_sanity_checks(stepped)
    return stepped


def integrate(initial: state.ModelState, nsteps: int):
    def _step(carry, step_idx):
        prev_state, curr_state = carry
        time_seconds = step_idx * cfg.dt
        raw_new_state = step(curr_state, time_seconds=time_seconds)

        if cfg.time_filter == "raw":
            filt_curr, filt_new = _robert_asselin_williams(prev_state, curr_state, raw_new_state)
            return (filt_curr, filt_new), filt_new
        if cfg.time_filter == "asselin":
            filt_curr, filt_new = _robert_asselin(prev_state, curr_state, raw_new_state)
            return (filt_curr, filt_new), filt_new

        # no filtering
        return (curr_state, raw_new_state), raw_new_state

    step_indices = jnp.arange(nsteps)
    carry_out, states = jax.lax.scan(_step, (initial, initial), step_indices)
    return carry_out[1], states


def _robert_asselin(
    prev_state: state.ModelState, curr_state: state.ModelState, new_state: state.ModelState
):
    """Classic Robert–Asselin filter to damp the leapfrog computational mode.

    The filter updates the *current* state using the two-time-level curvature
    ``prev - 2 * curr + new`` while leaving the newly stepped state unchanged.
    The filtered current state is passed forward as ``prev`` on the next
    timestep, providing the intended damping of the ±1 oscillation.
    """

    eps = cfg.ra

    filt_curr = curr_state.__class__(
        curr_state.zeta + eps * (prev_state.zeta - 2 * curr_state.zeta + new_state.zeta),
        curr_state.div + eps * (prev_state.div - 2 * curr_state.div + new_state.div),
        curr_state.T + eps * (prev_state.T - 2 * curr_state.T + new_state.T),
        curr_state.lnps + eps * (prev_state.lnps - 2 * curr_state.lnps + new_state.lnps),
    )

    return filt_curr, new_state

def _runtime_sanity_checks(mstate: state.ModelState):
    """Abort early if temperatures or pressures leave reasonable bounds,
    or just log them when AFES_VENUS_JAX_STRICT_SANITY=0.
    """
    import afes_venus_jax.spharm as sph

    T_grid = sph.synthesis_spec_to_grid(mstate.T, cfg.nlat, cfg.nlon)
    lnps_grid = sph.synthesis_spec_to_grid(mstate.lnps, cfg.nlat, cfg.nlon)
    ps_grid = cfg.ps_ref * jnp.exp(lnps_grid)

    # Per‑level mins/maxes – very useful for seeing if only top/bottom blow up
    T_min_levels = jnp.min(T_grid, axis=(-1, -2))
    T_max_levels = jnp.max(T_grid, axis=(-1, -2))

    jax.debug.print(
        "[sanity] T global: {mn} .. {mx}",
        mn=jnp.min(T_grid),
        mx=jnp.max(T_grid),
    )
    jax.debug.print(
        "[sanity] T level mins: {mins}",
        mins=T_min_levels,
    )
    jax.debug.print(
        "[sanity] T level maxs: {maxs}",
        maxs=T_max_levels,
    )
    jax.debug.print(
        "[sanity] ps global: {mn} .. {mx}",
        mn=jnp.min(ps_grid),
        mx=jnp.max(ps_grid),
    )

    # Original bounds
    max_T = float(jnp.max(T_grid))
    min_T = float(jnp.min(T_grid))
    max_ps = float(jnp.max(ps_grid))
    min_ps = float(jnp.min(ps_grid))

    if not STRICT_SANITY:
        # Just print warnings, don't abort
        if min_T < 100.0 or max_T > 1000.0:
            print(f"[WARN] Temperature bounds exceeded: min={min_T:.2f}, max={max_T:.2f}")
        if min_ps < 1e3 or max_ps > 1e7:
            print(f"[WARN] Surface pressure bounds exceeded: min={min_ps:.2e}, max={max_ps:.2e}")
        return

    if min_T < 100.0 or max_T > 1000.0:
        raise FloatingPointError(f"Temperature left bounds: min={min_T:.2f}, max={max_T:.2f}")
    if min_ps < 1e3 or max_ps > 1e7:
        raise FloatingPointError(f"Surface pressure left bounds: min={min_ps:.2e}, max={max_ps:.2e}")


def _robert_asselin_williams(
    prev_state: state.ModelState, curr_state: state.ModelState, new_state: state.ModelState
):
    eps = cfg.ra
    gamma = cfg.ra_williams_factor

    filt_curr = curr_state.__class__(
        curr_state.zeta + eps * (prev_state.zeta - 2 * curr_state.zeta + new_state.zeta),
        curr_state.div + eps * (prev_state.div - 2 * curr_state.div + new_state.div),
        curr_state.T + eps * (prev_state.T - 2 * curr_state.T + new_state.T),
        curr_state.lnps + eps * (prev_state.lnps - 2 * curr_state.lnps + new_state.lnps),
    )

    filt_new = new_state.__class__(
        new_state.zeta + 0.5 * gamma * (filt_curr.zeta - curr_state.zeta),
        new_state.div + 0.5 * gamma * (filt_curr.div - curr_state.div),
        new_state.T + 0.5 * gamma * (filt_curr.T - curr_state.T),
        new_state.lnps + 0.5 * gamma * (filt_curr.lnps - curr_state.lnps),
    )

    return filt_curr, filt_new
