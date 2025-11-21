import jax
import jax.numpy as jnp

import afes_venus_jax.config as cfg
import afes_venus_jax.spharm as sph


def test_roundtrip():
    key = jax.random.PRNGKey(0)
    grid = jax.random.normal(key, (cfg.nlat, cfg.nlon))
    spec = sph.analysis_grid_to_spec(grid)
    recon = sph.synthesis_spec_to_grid(spec, cfg.nlat, cfg.nlon)
    rel = jnp.linalg.norm(recon - grid) / jnp.linalg.norm(grid)
    assert rel < 1e-8


def test_laplacian_eigen():
    ell = 3
    m = 2
    spec = jnp.zeros((cfg.Lmax + 1, cfg.Lmax + 1), dtype=jnp.complex128)
    spec = spec.at[ell, m].set(1.0)
    lap = sph.lap_spec(spec)
    factor = -ell * (ell + 1) / (cfg.a ** 2)
    assert jnp.isclose(lap[ell, m], factor)
