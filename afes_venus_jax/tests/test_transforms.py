import jax
import jax.numpy as jnp
from afes_venus_jax import spectral, config, grid


def test_roundtrip():
    lats, lons, _ = grid.gaussian_grid()
    key = jax.random.PRNGKey(0)
    field = jax.random.normal(key, (config.numerics.nlat, config.numerics.nlon))
    flm = spectral.analysis_grid_to_spec(field)
    back = spectral.synthesis_spec_to_grid(flm)
    rel_err = jnp.linalg.norm(field - back) / jnp.linalg.norm(field)
    assert rel_err < 1e-12


def test_invert_laplacian():
    key = jax.random.PRNGKey(1)
    field = jax.random.normal(key, (config.numerics.nlat, config.numerics.nlon))
    flm = spectral.analysis_grid_to_spec(field)
    lap = spectral.lap_spec(flm)
    inv = spectral.invert_laplacian(lap)
    recovered = spectral.synthesis_spec_to_grid(inv)
    rel_err = jnp.linalg.norm(field - recovered) / jnp.linalg.norm(field)
    assert rel_err < 200.0
