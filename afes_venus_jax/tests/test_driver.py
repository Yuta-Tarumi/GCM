from afes_venus_jax import driver


def test_driver_run_returns_dataset():
    ds = driver.run(nsteps=3, history_stride=1)
    assert "zeta" in ds
    assert ds.dims["lat"] > 0 and ds.dims["lon"] > 0
    assert ds.dims["time"] >= 1
