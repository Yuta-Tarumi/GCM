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


def test_balanced_spinup_initial_condition_has_small_zonal_mean():
    """Random winds should not contain strong zonal-mean bands."""

    mstate = balanced_example.balanced_random_initial_condition(seed=0, wind_std=5.0)

    psi, chi = balanced_example.sph.psi_chi_from_zeta_div(mstate.zeta, mstate.div)
    u, v = balanced_example.sph.uv_from_psi_chi(
        psi, chi, balanced_example.cfg.nlat, balanced_example.cfg.nlon
    )

    u_zonal_mean = u.mean(axis=-1)
    v_zonal_mean = v.mean(axis=-1)

    assert (abs(u_zonal_mean) < 0.5).all()
    assert (abs(v_zonal_mean) < 0.5).all()

