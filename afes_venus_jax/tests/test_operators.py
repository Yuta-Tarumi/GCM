import jax
import jax.numpy as jnp
from afes_venus_jax.spharm import (
    analysis_grid_to_spec,
    psi_chi_from_vort_div,
    uv_from_psi_chi,
    vort_div_from_uv,
)
from afes_venus_jax.config import DEFAULT_CFG
from afes_venus_jax.grid import gaussian_grid


def test_vort_div_consistency():
    cfg = DEFAULT_CFG
    zeta = jnp.zeros((cfg.Lmax + 1, cfg.Lmax + 1), dtype=jnp.complex128).at[3, 2].set(1.0)
    div = jnp.zeros_like(zeta)
    psi, chi = psi_chi_from_vort_div(zeta, div, cfg)
    u, v = uv_from_psi_chi(psi, chi, cfg)
    vort_back, div_back = vort_div_from_uv(u, v, cfg)
    rel_vort = jnp.linalg.norm(vort_back - zeta) / jnp.linalg.norm(zeta)
    assert rel_vort < 5e-3
    assert jnp.linalg.norm(div_back) < 5e-3


def test_solid_body_longitude_independence():
    """A longitude-free streamfunction should yield longitude-free winds."""

    cfg = DEFAULT_CFG
    lats, _, _ = gaussian_grid(cfg)

    # Streamfunction for solid-body rotation about the polar axis.
    psi_lat = -cfg.a**2 * (0.5 * lats + 0.25 * jnp.sin(2 * lats))
    psi_grid = psi_lat[:, None] * jnp.ones((cfg.nlat, cfg.nlon))
    psi_spec = analysis_grid_to_spec(psi_grid, cfg)
    chi_spec = jnp.zeros_like(psi_spec)
    u, v = uv_from_psi_chi(psi_spec, chi_spec, cfg)

    zeta_spec, div_spec = vort_div_from_uv(u, v, cfg)

    assert jnp.max(jnp.std(u, axis=1)) < 1e-10
    assert jnp.max(jnp.std(v, axis=1)) < 1e-10
    assert jnp.max(jnp.abs(zeta_spec[..., 1:])) < 1e-10
    assert jnp.max(jnp.abs(div_spec[..., 1:])) < 1e-10
