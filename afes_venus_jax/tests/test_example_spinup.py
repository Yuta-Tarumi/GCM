import afes_venus_jax.examples.t42l60_venus_dry_spinup as spinup


def test_initial_condition_passes_sanity_check():
    mstate = spinup.initial_condition()
    spinup.sanity_check_initial_condition(mstate)
