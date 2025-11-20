import jax
import jax.numpy as jnp
from afes_venus_jax.spharm import analysis_grid_to_spec, synthesis_spec_to_grid, lap_spec
from afes_venus_jax.config import DEFAULT_CFG


def test_roundtrip():
    cfg = DEFAULT_CFG
    spec = jnp.zeros((cfg.Lmax + 1, cfg.Lmax + 1), dtype=jnp.complex128)
    spec = spec.at[3, 2].set(1.0)

    grid = synthesis_spec_to_grid(spec, cfg)
    spec_back = analysis_grid_to_spec(grid, cfg)
    rel_err = jnp.linalg.norm(spec - spec_back) / jnp.linalg.norm(spec)
    assert rel_err < 1e-10


def test_laplacian_eigen():
    cfg = DEFAULT_CFG
    key = jax.random.PRNGKey(1)
    grid = jax.random.normal(key, (cfg.nlat, cfg.nlon), dtype=jnp.float64)
    ell = 3
    m = 2
    spec = jnp.zeros((cfg.Lmax + 1, cfg.Lmax + 1), dtype=jnp.complex128)
    spec = spec.at[ell, m].set(1.0)

    lap_spec_vals = lap_spec(spec, cfg)
    y_lm = synthesis_spec_to_grid(spec, cfg)
    lap_grid = synthesis_spec_to_grid(lap_spec_vals, cfg)

    eigenvalue = -(ell * (ell + 1) / cfg.a**2)
    expected = eigenvalue * y_lm
    rel = jnp.linalg.norm(lap_grid - expected) / jnp.linalg.norm(expected)
    assert rel < 1e-10
