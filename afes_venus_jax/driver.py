"""High-level run driver and convenience utilities."""
from __future__ import annotations
from pathlib import Path
import xarray as xr
from . import config, state
from .dynamics import integrators
from . import io


def run(
    cfg: config.ModelConfig | None = None,
    nsteps: int = 10,
    history_stride: int = 6,
    initial_state: state.ModelState | None = None,
) -> xr.Dataset:
    if cfg is None:
        cfg = config.DEFAULT
    if initial_state is None:
        initial_state = state.initial_isothermal()
    cur = initial_state
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


def run_afes_venus_paper_demo(
    nsteps: int = 240,
    history_stride: int = 12,
    angular_speed: float | None = None,
    equatorial_speed: float | None = -100.0,
    base_temperature: float = 240.0,
) -> xr.Dataset:
    """Replicate the AFES-Venus paper settings with a solid-body initial wind.

    The configuration matches the hyperdiffusion, vertical diffusion and
    Rayleigh damping described in Sugimoto et al. (2023). The initial condition
    follows their solid-body westward wind, optionally scaled via
    ``angular_speed``.
    """

    cfg_path = Path(__file__).with_name("config").joinpath("afes_venus_paper.yaml")
    cfg = config.load_config(cfg_path)
    initial = state.initial_solid_body_rotation(
        angular_speed=angular_speed,
        equatorial_speed=equatorial_speed,
        base_temperature=base_temperature,
        ps=cfg.planet.surface_pressure,
        L=cfg.numerics.nlev,
        nlat=cfg.numerics.nlat,
        nlon=cfg.numerics.nlon,
    )
    return run(cfg=cfg, nsteps=nsteps, history_stride=history_stride, initial_state=initial)
