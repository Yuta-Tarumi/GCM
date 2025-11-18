import numpy as np

from afes_venus_jax.examples import run_and_movie


def test_vortex_pair_example_produces_bounded_fields():
    frames, lats, lons = run_and_movie.run_example(nsteps=5, scenario="vortex_pair")
    assert len(frames) == 5
    stack = np.stack(frames)
    assert np.all(np.isfinite(stack))
    assert np.ptp(stack[0]) > 0
    assert lats.shape[0] > 0 and lons.shape[0] > 0
