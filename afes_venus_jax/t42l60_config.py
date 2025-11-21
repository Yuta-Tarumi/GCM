"""AFES-like presets for the T42L60 Venus dry spin-up configuration."""
from __future__ import annotations

import jax.numpy as jnp

from dataclasses import dataclass

# Resolution
LMAX = 42
NLAT = 64
NLON = 128
LLEVELS = 60

# Time stepping
DT = 150.0  # [s] conservative for stability in AFES-like setup
SI_ALPHA = 0.5
TIME_FILTER = "asselin"  # alternatives: "raw"
RA_COEFF = 0.08  # plain Asselin strength for leapfrog
RA_WILLIAMS_FACTOR = 0.53

# Hyperdiffusion tuned for T42: tau at truncation ~0.01 Earth days (~864 s)
HYPERDIFF_TAU_KMAX = 0.01 * 86400.0
TAU_HYPERDIFF = HYPERDIFF_TAU_KMAX
NU4_HYPERDIFF = None  # computed on the fly when needed

# Divergence damping (weak) to keep gravity wave noise in check
TAU_DIV_DAMP = 0.5 * 86400.0

# Vertical diffusion (eddy viscosity)
KZ = 0.15  # m^2/s applied to u, v, T

# Lower-boundary Rayleigh drag
TAU_BOTTOM = 0.5 * 86400.0
BOTTOM_LAYERS_RAMP = 3  # linearly relax over lowest three levels

# Upper sponge configuration
@dataclass(frozen=True)
class SpongeConfig:
    top_levels: int
    tau_min: float
    tau_base: float
    apply_to: tuple[str, ...]


SPONGE = SpongeConfig(
    top_levels=10,  # damp roughly upper 10 levels (>~80 km for sigma grid)
    tau_min=0.1 * 86400.0,
    tau_base=3.0 * 86400.0,
    apply_to=("u", "v", "T"),
)


# Newtonian cooling profile (simple Venus-like)
sigma_full = jnp.linspace(0.5 / LLEVELS, 1.0 - 0.5 / LLEVELS, LLEVELS)
T_EQ_BOTTOM = 735.0
T_EQ_TOP = 170.0
T_EQ_PROFILE = T_EQ_BOTTOM - (T_EQ_BOTTOM - T_EQ_TOP) * (1.0 - sigma_full) ** 0.8

# Relaxation times: very slow near surface, fast aloft (log-linear in sigma)
TAU_RAD_SURFACE = 10_000.0 * 86400.0
TAU_RAD_TOP = 0.1 * 86400.0
TAU_RAD_PROFILE = TAU_RAD_SURFACE * (TAU_RAD_TOP / TAU_RAD_SURFACE) ** (1.0 - sigma_full)
