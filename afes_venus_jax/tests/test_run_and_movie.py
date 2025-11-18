import numpy as np
import jax

from afes_venus_jax.examples import run_and_movie


def _preferred_device():
    devices = jax.devices()
    for dev in devices:
        if dev.platform == "gpu":
            return dev
    return devices[0]


def test_vortex_pair_example_produces_bounded_fields():
    frames, lats, lons = run_and_movie.run_example(nsteps=5, scenario="vortex_pair")
    assert len(frames) == 5
    stack = np.stack(frames)
    assert np.all(np.isfinite(stack))
    assert np.ptp(stack[0]) > 0
    assert lats.shape[0] > 0 and lons.shape[0] > 0


def test_vortex_pair_moves_when_run_on_fastest_device():
    device = _preferred_device()
    frames, _, _ = run_and_movie.run_example(nsteps=8, scenario="vortex_pair", device=device)
    stack = np.stack(frames)
    assert stack.shape[0] == 8
    # Motion of the balanced vortices should yield measurable differences
    # between early and late frames (tuned empirically for both CPU and GPU).
    delta = np.mean(np.abs(stack[-1] - stack[0]))
    assert delta > 1e-8
    temporal_variation = np.std(np.diff(stack, axis=0))
    assert temporal_variation > 1e-9
