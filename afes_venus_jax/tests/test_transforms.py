import jax
import jax.numpy as jnp
from afes_venus_jax.spharm import analysis_grid_to_spec, synthesis_spec_to_grid, lap_spec
from afes_venus_jax.config import DEFAULT_CFG


def test_roundtrip():
    cfg = DEFAULT_CFG
    key = jax.random.PRNGKey(0)
    grid = jax.random.normal(key, (cfg.nlat, cfg.nlon), dtype=jnp.float64)
    spec = analysis_grid_to_spec(grid)
    back = synthesis_spec_to_grid(spec, cfg.nlat, cfg.nlon)
    rel_err = jnp.linalg.norm(grid - back) / jnp.linalg.norm(grid)
    assert rel_err < 1e-8


def test_laplacian_eigen():
    cfg = DEFAULT_CFG
    key = jax.random.PRNGKey(1)
    grid = jax.random.normal(key, (cfg.nlat, cfg.nlon), dtype=jnp.float64)
    spec = analysis_grid_to_spec(grid)
    lap_spec_vals = lap_spec(spec, cfg)
    lap_grid = synthesis_spec_to_grid(lap_spec_vals, cfg.nlat, cfg.nlon)
    d2 = jnp.gradient(jnp.gradient(grid, axis=-1), axis=-1) + jnp.gradient(jnp.gradient(grid, axis=-2), axis=-2)
    rel = jnp.linalg.norm(lap_grid - d2) / jnp.linalg.norm(d2)
    assert rel < 1.5
