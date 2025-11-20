import numpy as np
import jax

from afes_venus_jax import config
from afes_venus_jax.examples import superrotation_demo


def _preferred_device():
    for dev in jax.devices():
        if dev.platform == "gpu":
            return dev
    return jax.devices()[0]


def test_solid_body_superrotation_remains_prograde():
    device = _preferred_device()
    cfg = config.DEFAULT
    initial = superrotation_demo.initial_solid_body_superrotation(u_equator=70.0, cfg=cfg)
    initial = jax.tree_util.tree_map(lambda arr: jax.device_put(arr, device), initial)

    frames, profiles, _, _, _, heights = superrotation_demo.collect_superrotation_history(
        nsteps=6,
        sample_interval=3,
        seed=0,
        equatorial_half_width=20.0,
        initial_state=initial,
        device=device,
        cfg=cfg,
    )

    assert len(frames) == 3
    initial_profile = profiles[0]
    final_profile = profiles[-1]
    target_idx = int(np.argmin(np.abs((heights / 1000.0) - 60.0)))

    assert initial_profile[target_idx] > 0.0
    assert final_profile[target_idx] > 0.5 * initial_profile[target_idx]
    assert np.max(final_profile) > 10.0
