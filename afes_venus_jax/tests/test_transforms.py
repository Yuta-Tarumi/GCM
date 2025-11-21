import jax
import jax.numpy as jnp

from afes_venus_jax.config import ModelConfig
from afes_venus_jax.spharm import analysis_grid_to_spec, synthesis_spec_to_grid, lap_spec, spectral_shape


def test_round_trip():
    cfg = ModelConfig(Lmax=6, nlat=16, nlon=32, L=4)
    key = jax.random.PRNGKey(0)
    spec = jax.random.normal(key, spectral_shape(cfg)) + 1j * jax.random.normal(key, spectral_shape(cfg))
    grid = synthesis_spec_to_grid(spec, cfg)
    spec_back = analysis_grid_to_spec(grid, cfg)
    rel_err = jnp.linalg.norm(spec - spec_back) / jnp.linalg.norm(spec)
    assert rel_err < 1.0


def test_laplacian_eigen():
    cfg = ModelConfig(Lmax=6, nlat=16, nlon=32, L=4)
    spec = jnp.zeros(spectral_shape(cfg), dtype=jnp.complex128)
    spec = spec.at[3, cfg.Lmax + 1].set(1.0 + 0j)
    lap_spec_field = lap_spec(spec, cfg)
    spec_back = analysis_grid_to_spec(synthesis_spec_to_grid(spec, cfg), cfg)
    err = jnp.max(jnp.abs(lap_spec_field - lap_spec(spec_back, cfg)))
    assert err < 1e-6
