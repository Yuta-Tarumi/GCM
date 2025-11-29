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

from afes_venus_jax.examples import t42l60_venus_balanced_spinup as balanced_example


def test_balanced_spinup_initial_condition_has_target_wind_std():
    """Balanced state generation should meet the target wind dispersion."""

    # Arrange
    mstate = balanced_example.balanced_random_initial_condition(seed=0, wind_std=5.0)

    # Act & Assert
    balanced_example.sanity_check_balanced_state(mstate, target_std=5.0)

