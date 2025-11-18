import numpy as np
import jax.numpy as jnp
from afes_venus_jax import config, state, spectral
from afes_venus_jax.physics import solar, newtonian


def test_solar_heating_positive():
    cfg = config.DEFAULT
    heating = solar.solar_heating_tendency(time_s=0.0, cfg=cfg)
    heating_grid = spectral.synthesis_spec_to_grid(heating)
    assert heating_grid.shape[0] == cfg.numerics.nlev
    assert jnp.all(heating_grid >= -1e-12)


def test_newtonian_cooling_targets_profile():
    cfg = config.DEFAULT
    s = state.initial_isothermal(T0=400.0)
    tendency = newtonian.cooling_tendency(s, cfg)
    grid_tend = spectral.synthesis_spec_to_grid(tendency)
    assert np.mean(grid_tend) < 0.0
