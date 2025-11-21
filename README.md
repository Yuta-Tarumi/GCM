# AFES-Venus-JAX

A lightweight demonstration of a hydrostatic primitive-equation spectral
core in JAX using a sigma–Lorenz vertical grid. The package mirrors the
layout of an AFES-Venus style core with spherical-harmonic transforms,
semi-implicit stepping, and spectral hyperdiffusion.

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
  explicit quadrature for clarity and JIT compatibility.
- Vertical sigma levels follow an exponential mapping from altitude with
  a reference scale height of 15 km.
- Hyperdiffusion uses a configurable ∇⁴ operator with an e-folding time
  at truncation wavenumber.
