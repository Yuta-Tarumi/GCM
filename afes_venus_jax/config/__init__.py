"""Model configuration helpers and defaults."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping
import math
try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback path
    yaml = None
import jax.numpy as jnp

__all__ = [
    "PlanetConfig",
    "NumericsConfig",
    "PhysicsConfig",
    "ModelConfig",
    "load_config",
    "DEFAULT",
    "spectral_wavenumbers",
]


@dataclass(frozen=True)
class PlanetConfig:
    """Planetary constants and orbital parameters."""

    radius: float
    gravity: float
    rotation_rate: float
    solar_day_rate: float
    solar_constant: float
    obliquity_deg: float
    surface_pressure: float


@dataclass(frozen=True)
class NumericsConfig:
    """Discrete resolution and time stepping settings."""

    truncation: int
    nlat: int
    nlon: int
    nlev: int
    dt: float
    semi_implicit_alpha: float
    hyperdiff_order: int
    hyperdiff_tau_smallest: float
    dealiasing: bool
    leapfrog_raw_coeff: float


@dataclass(frozen=True)
class PhysicsConfig:
    """Physical parameterizations toggles."""

    solar_profile: str
    newtonian_profile: str
    enable_solar: bool
    enable_newtonian: bool
    solar_peak_heating_K_per_day: float
    kz: float
    rayleigh_low_tau: float
    sponge_top_height: float
    sponge_tau: float


@dataclass(frozen=True)
class ModelConfig:
    """Complete model configuration tree."""

    planet: PlanetConfig
    numerics: NumericsConfig
    physics: PhysicsConfig

    @property
    def rotation_period(self) -> float:
        return 2 * math.pi / abs(self.planet.rotation_rate)


def _resolve_yaml(path: Path | str | None) -> Path:
    if path is None:
        return Path(__file__).with_name("venus_t42l60.yaml")
    return Path(path)


def _convert_scalar(value: str) -> Any:
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _simple_yaml_parse(text: str) -> MutableMapping[str, Any]:
    root: MutableMapping[str, Any] = {}
    stack: list[tuple[int, MutableMapping[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        key, _, rest = raw.strip().partition(":")
        value = rest.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not value:
            new_dict: MutableMapping[str, Any] = {}
            parent[key] = new_dict
            stack.append((indent, new_dict))
        else:
            parent[key] = _convert_scalar(value)
    return root


def _load_yaml(path: Path) -> MutableMapping[str, Any]:
    text = path.read_text()
    if yaml is not None:
        return yaml.safe_load(text)
    return _simple_yaml_parse(text)


def _planet_from_dict(data: Mapping[str, Any]) -> PlanetConfig:
    rotation_rate = data.get("rotation_rate")
    if rotation_rate is None:
        retrograde = data.get("retrograde", True)
        period = float(data.get("rotation_period_days", 116.75))
        rotation_rate = (-1 if retrograde else 1) * 2 * math.pi / (period * 86400.0)
    orbital_period = data.get("orbital_period_days")
    if orbital_period is None:
        solar_day_rate = rotation_rate
    else:
        orbital_rate = 2 * math.pi / (float(orbital_period) * 86400.0)
        solar_day_rate = rotation_rate - orbital_rate
    return PlanetConfig(
        radius=float(data["radius"]),
        gravity=float(data["gravity"]),
        rotation_rate=float(rotation_rate),
        solar_day_rate=float(solar_day_rate),
        solar_constant=float(data["solar_constant"]),
        obliquity_deg=float(data.get("obliquity_deg", 2.64)),
        surface_pressure=float(data.get("surface_pressure", 9.2e6)),
    )


def _numerics_from_dict(data: Mapping[str, Any]) -> NumericsConfig:
    return NumericsConfig(
        truncation=int(data.get("truncation", 42)),
        nlat=int(data.get("nlat", 64)),
        nlon=int(data.get("nlon", 128)),
        nlev=int(data.get("nlev", 60)),
        dt=float(data.get("dt", 1200.0)),
        semi_implicit_alpha=float(data.get("semi_implicit_alpha", 0.5)),
        hyperdiff_order=int(data.get("hyperdiff_order", 4)),
        hyperdiff_tau_smallest=float(data.get("hyperdiff_tau_smallest", 0.1 * 86400.0)),
        dealiasing=bool(data.get("dealiasing", True)),
        leapfrog_raw_coeff=float(data.get("leapfrog_raw_coeff", 0.04)),
    )


def _physics_from_dict(data: Mapping[str, Any]) -> PhysicsConfig:
    return PhysicsConfig(
        solar_profile=data.get("solar_profile", "uniform_50_80km"),
        newtonian_profile=data.get("newtonian_profile", "crisp1989"),
        enable_solar=bool(data.get("enable_solar", True)),
        enable_newtonian=bool(data.get("enable_newtonian", True)),
        solar_peak_heating_K_per_day=float(data.get("solar_peak_heating_K_per_day", 12.0)),
        kz=float(data.get("kz", 0.15)),
        rayleigh_low_tau=float(data.get("rayleigh_low_tau", 86400.0)),
        sponge_top_height=float(data.get("sponge_top_height", 80_000.0)),
        sponge_tau=float(data.get("sponge_tau", 43200.0)),
    )


def load_config(path: Path | str | None = None) -> ModelConfig:
    data = _load_yaml(_resolve_yaml(path))
    planet = _planet_from_dict(data["planet"])
    numerics = _numerics_from_dict(data["numerics"])
    physics = _physics_from_dict(data["physics"])
    return ModelConfig(planet=planet, numerics=numerics, physics=physics)


DEFAULT = load_config()
"""Default Venus T42/L60 configuration."""


def spectral_wavenumbers(lmax: int | None = None, radius: float | None = None) -> jnp.ndarray:
    """Return squared total wavenumber array for triangular truncation."""
    cfg = DEFAULT
    if lmax is None:
        lmax = cfg.numerics.truncation
    if radius is None:
        radius = cfg.planet.radius
    l = jnp.arange(lmax + 1)
    return l * (l + 1) / (radius ** 2)


# Backwards-compatible module-level shortcuts
planet = DEFAULT.planet
numerics = DEFAULT.numerics
physics = DEFAULT.physics

# Frequently used scalars for convenience
a = planet.radius
g = planet.gravity
Omega = planet.rotation_rate
ps_ref = planet.surface_pressure
cp = 1000.0
R_gas = 8.314462618 / 0.04401
Lmax = numerics.truncation
nlat = numerics.nlat
nlon = numerics.nlon
L = numerics.nlev
dt = numerics.dt
alpha = numerics.semi_implicit_alpha
ra = physics.rayleigh_low_tau
order_hdiff = numerics.hyperdiff_order
tau_hdiff = numerics.hyperdiff_tau_smallest
