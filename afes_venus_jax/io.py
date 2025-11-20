"""Minimal NetCDF I/O helpers using optional xarray."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from . import config, grid, spectral
from .state import ModelState

try:
    import xarray as xr  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback
    xr = None


class SimpleDataset:
    """Fallback container exposing a subset of the xarray Dataset API."""

    def __init__(self, data_vars: dict[str, np.ndarray], coords: dict[str, np.ndarray]):
        self.data_vars = data_vars
        self.coords = coords
        self.dims = {name: len(np.atleast_1d(values)) for name, values in coords.items()}

    def __contains__(self, key: str) -> bool:  # pragma: no cover - trivial
        return key in self.data_vars

    def __getitem__(self, key: str) -> np.ndarray:  # pragma: no cover - debug helper
        return self.data_vars[key]


def _state_to_arrays(state: ModelState):
    return {
        "zeta": np.asarray(spectral.synthesis_spec_to_grid(state.zeta)),
        "div": np.asarray(spectral.synthesis_spec_to_grid(state.div)),
        "T": np.asarray(spectral.synthesis_spec_to_grid(state.T)),
        "lnps": np.asarray(spectral.synthesis_spec_to_grid(state.lnps)),
    }


def as_dataset(state: ModelState, cfg: config.ModelConfig | None = None):
    if cfg is None:
        cfg = config.DEFAULT
    arrays = _state_to_arrays(state)
    levels = np.arange(cfg.numerics.nlev)
    lats, lons, _ = grid.gaussian_grid(cfg.numerics.nlat, cfg.numerics.nlon)
    coords = {
        "level": np.asarray(levels),
        "lat": np.asarray(lats),
        "lon": np.asarray(lons),
    }
    if xr is None:
        return SimpleDataset(arrays, coords)
    data_vars = {
        "zeta": ("level", "lat", "lon", arrays["zeta"]),
        "div": ("level", "lat", "lon", arrays["div"]),
        "T": ("level", "lat", "lon", arrays["T"]),
        "lnps": ("lat", "lon", arrays["lnps"]),
    }
    return xr.Dataset(data_vars=data_vars, coords=coords)


def history_dataset(states: list[ModelState], times: list[float], cfg: config.ModelConfig | None = None):
    if cfg is None:
        cfg = config.DEFAULT
    arrays = [_state_to_arrays(s) for s in states]
    lat, lon, _ = grid.gaussian_grid(cfg.numerics.nlat, cfg.numerics.nlon)
    level = np.arange(cfg.numerics.nlev)
    coords = {
        "time": np.asarray(times),
        "level": np.asarray(level),
        "lat": np.asarray(lat),
        "lon": np.asarray(lon),
    }
    data_vars = {
        "zeta": np.stack([a["zeta"] for a in arrays], axis=0),
        "div": np.stack([a["div"] for a in arrays], axis=0),
        "T": np.stack([a["T"] for a in arrays], axis=0),
        "lnps": np.stack([a["lnps"] for a in arrays], axis=0),
    }
    if xr is None:
        return SimpleDataset(data_vars, coords)
    xr_data = {
        "zeta": (("time", "level", "lat", "lon"), data_vars["zeta"]),
        "div": (("time", "level", "lat", "lon"), data_vars["div"]),
        "T": (("time", "level", "lat", "lon"), data_vars["T"]),
        "lnps": (("time", "lat", "lon"), data_vars["lnps"]),
    }
    return xr.Dataset(data_vars=xr_data, coords=coords)


def write_checkpoint(state: ModelState, path: str | Path, cfg: config.ModelConfig | None = None) -> Path:
    if xr is None:
        raise ImportError("xarray is required for NetCDF output")
    ds = as_dataset(state, cfg)
    out = Path(path)
    ds.to_netcdf(out)
    return out
