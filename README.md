# AFES-Venus-JAX

A lightweight demonstration of a hydrostatic primitive-equation spectral
core in JAX using a sigma–Lorenz vertical grid. The package mirrors the
layout of an AFES-Venus style core with spherical-harmonic transforms,
semi-implicit stepping, optional semi-Lagrangian advection, divergence
damping, Robert–Asselin filtering, and spectral hyperdiffusion. The
defaults now target a single AFES-like configuration: **T42L60 Venus dry
spin-up** with conservative 150 s time steps.

## Installation

```bash
pip install -e .
```

To run the test suite on resource-limited machines, set lower-resolution
defaults via environment variables before invoking pytest:

```bash
export AFES_VENUS_JAX_LMAX=10
export AFES_VENUS_JAX_NLAT=32
export AFES_VENUS_JAX_NLON=64
export AFES_VENUS_JAX_L=10
export AFES_VENUS_JAX_FAST_TESTS=1
pytest
```

## Running the T42L60 demo

```bash
python -m afes_venus_jax.examples.t42l60_venus_dry_spinup
```

## Notes

- The only supported/tested setup is the AFES-like T42L60 Venus dry
  configuration. A dedicated ``afes_venus_jax.t42l60_config`` module
  lists the numerical knobs (hyperdiffusion 0.01 day e-folding at T42,
  Kz=0.15 m²/s, bottom Rayleigh drag 0.5 day, upper sponge, and a
  Newtonian cooling profile).
- Time stepping uses leapfrog with the plain Robert–Asselin filter by
  default (α≈0.08). Set ``AFES_VENUS_JAX_TIME_FILTER=raw`` to switch to
  RAW if desired.
- Vertical sigma levels follow an exponential mapping from altitude with
  a reference scale height of 15 km.
- Hyperdiffusion uses a ∇⁴ operator with coefficients derived from the
  requested e-folding time at truncation.
