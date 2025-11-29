"""Regression test guarding against early temperature blow-ups in the T42L60 demo.

The T42L60 Venus dry spin-up previously raised ``FloatingPointError`` from
``timestep._runtime_sanity_checks`` around step 5 when temperatures left the
100–1000 K bounds. This test runs a short integration to ensure temperatures
remain within those runtime limits.
"""
import os

import pytest

# Disable JIT by default to keep the short integration lightweight in CI.
os.environ.setdefault("JAX_DISABLE_JIT", "1")

import afes_venus_jax.config as cfg
import afes_venus_jax.spharm as sph
from afes_venus_jax.examples import t42l60_venus_dry_spinup as example
import afes_venus_jax.timestep as timestep


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("AFES_VENUS_JAX_FAST_TESTS") in {"1", "true", "True"},
    reason="Skip heavy T42L60 integration when fast tests are requested.",
)
def test_t42l60_spinup_temperature_stays_within_runtime_bounds():
    """Advance a few steps and confirm temperatures stay in the runtime bounds."""

    # Arrange: build the same initial condition used by the demo spin-up.
    mstate = example.initial_condition()
    example.sanity_check_initial_condition(mstate)
    dt = cfg.dt

    # Act: run a handful of steps to expose the early-step blow-up if present.
    time_seconds = 0.0
    for _ in range(8):
        mstate = timestep.step(mstate, time_seconds=time_seconds)
        time_seconds += dt

    # Assert: temperatures respect the runtime sanity check thresholds.
    timestep._runtime_sanity_checks(mstate)
    T_grid = sph.synthesis_spec_to_grid(mstate.T, cfg.nlat, cfg.nlon)
    assert T_grid.min() >= 100.0
    assert T_grid.max() <= 1000.0
