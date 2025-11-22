import os

# Light-but-stable configuration for CI; disable JIT to avoid long compilation
os.environ.setdefault("JAX_DISABLE_JIT", "1")
# times in constrained test environments.
os.environ.setdefault("AFES_VENUS_JAX_LMAX", "4")
# Light-but-stable configuration for CI
os.environ.setdefault("AFES_VENUS_JAX_NLAT", "8")
os.environ.setdefault("AFES_VENUS_JAX_NLON", "16")
os.environ.setdefault("AFES_VENUS_JAX_L", "5")
os.environ.setdefault("AFES_VENUS_JAX_DT", "10.0")
os.environ.setdefault("AFES_VENUS_JAX_TAU_HDIFF", "200.0")
os.environ.setdefault("AFES_VENUS_JAX_KZ", "1.0")

import afes_venus_jax.config as cfg
from afes_venus_jax.examples import t42l60_venus_dry_spinup as example
import afes_venus_jax.timestep as timestep


def test_t42l60_venus_dry_spinup_stable_for_5_steps():
    # Arrange
    mstate = example.initial_condition()
    example.sanity_check_initial_condition(mstate)
    dt = cfg.dt

    # Act: advance a handful of steps. Any temperature blow-up should trigger
    # _runtime_sanity_checks inside ``timestep.step``.
    t = 0.0
    for _ in range(5):
        mstate = timestep.step(mstate, time_seconds=t)
        t += dt

    # Extra guard: ensure temperatures stay within the sanity bounds.
    # If the bounds are violated the loop above will already have raised.
    # Keeping this assertion makes the expectation explicit for the test.
    import afes_venus_jax.spharm as sph

    T_grid = sph.synthesis_spec_to_grid(mstate.T, cfg.nlat, cfg.nlon)
    assert T_grid.min() >= 100.0
    assert T_grid.max() <= 1000.0
