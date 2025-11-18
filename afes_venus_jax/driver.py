"""High-level run driver and convenience utilities."""
from __future__ import annotations
from . import config, state
from .dynamics import integrators
from . import io


def run(cfg: config.ModelConfig | None = None, nsteps: int = 10, history_stride: int = 6) -> xr.Dataset:
    if cfg is None:
        cfg = config.DEFAULT
    cur = state.initial_isothermal()
    history_states: list[state.ModelState] = []
    history_times: list[float] = []
    for step in range(nsteps):
        t = step * cfg.numerics.dt
        cur = integrators.step(cur, t, cfg)
        if step % history_stride == 0:
            history_states.append(cur)
            history_times.append(step * cfg.numerics.dt)
    if not history_states:
        history_states.append(cur)
        history_times.append(0.0)
    return io.history_dataset(history_states, history_times, cfg)
