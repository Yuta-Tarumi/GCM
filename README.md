# AFES-Venus-JAX

A lightweight demonstration of a hydrostatic primitive-equation spectral
core in JAX using a sigma–Lorenz vertical grid. The package mirrors the
layout of an AFES-Venus style core with spherical-harmonic transforms,
semi-implicit stepping, optional semi-Lagrangian advection, divergence
damping, Robert–Asselin–Williams filtering, and spectral hyperdiffusion.

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

- Gaussian grid and spherical-harmonic transforms are implemented by
  explicit quadrature for clarity and JIT compatibility. Set
  ``AFES_VENUS_JAX_USE_S2FFT=1`` to switch to the accelerated S2FFT
  equiangular sampling and optimised transforms used in AFES-style
  production builds.
- Enable ``AFES_VENUS_JAX_USE_SEMI_LAGRANGIAN_ADVECTION=1`` to advect
  winds, temperature, and surface pressure with a first-order
  semi-Lagrangian step. Use ``AFES_VENUS_JAX_USE_RAW_FILTER=0`` to fall
  back to the simpler Robert–Asselin filter.
- Vertical sigma levels follow an exponential mapping from altitude with
  a reference scale height of 15 km.
- Hyperdiffusion uses a configurable ∇⁴ operator with an e-folding time
  at truncation wavenumber.
