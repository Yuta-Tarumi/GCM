import os

import jax
import jax.numpy as jnp

import afes_venus_jax.config as cfg
import afes_venus_jax.grid as grid
import afes_venus_jax.spharm as sph


FAST = os.getenv("AFES_VENUS_JAX_FAST_TESTS", "0") == "1"


def test_roundtrip():
    key = jax.random.PRNGKey(0)
    grid = jax.random.normal(key, (cfg.nlat, cfg.nlon))
    spec = sph.analysis_grid_to_spec(grid)
    recon = sph.synthesis_spec_to_grid(spec, cfg.nlat, cfg.nlon)
    rel = jnp.linalg.norm(recon - grid) / jnp.linalg.norm(grid)
    tol = 1e-8 if not FAST else 1.0
    assert rel < tol


def test_laplacian_eigen():
    ell = 3
    m = 2
    spec = jnp.zeros((cfg.Lmax + 1, cfg.Lmax + 1), dtype=jnp.complex128)
    spec = spec.at[ell, m].set(1.0)
    lap = sph.lap_spec(spec)
    factor = -ell * (ell + 1) / (cfg.a ** 2)
    atol = 1e-12 if not FAST else 1e-4
    assert jnp.isclose(lap[ell, m], factor, atol=atol).item()


def test_uv_from_psi_chi_polar_behavior():
    lats, lons, _ = grid.gaussian_grid(cfg.nlat, cfg.nlon)
    lat2d, _ = jnp.meshgrid(jnp.array(lats), jnp.array(lons), indexing="ij")

    u0 = 50.0
    psi_grid = -cfg.a * u0 * jnp.sin(lat2d)
    psi_spec = sph.analysis_grid_to_spec(psi_grid)
    chi_spec = jnp.zeros_like(psi_spec)

    u, v = sph.uv_from_psi_chi(psi_spec, chi_spec, cfg.nlat, cfg.nlon)

    atol = 1e-2 if FAST else 1e-4
    assert jnp.allclose(v, 0.0, atol=atol)
    assert jnp.all(u > -1e-3)

    midband = slice(2, -2)
    assert jnp.allclose(u[midband], u0, rtol=1e-2, atol=1e-2)
    assert jnp.all(jnp.abs(u[(0, -1)]) < 1.0)
