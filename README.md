# AFES Venus JAX (minimal)

A light-weight hydrostatic primitive-equation core for Venus built on top of
[JAX](https://github.com/google/jax). This minimal snapshot retains only the
dynamics and physics core—the automated test suites have been removed. The
model follows the specification from the user story:

* T42 spectral truncation on a 128×64 Gaussian grid with 60 σ-levels.
* Prognostic variables: relative vorticity (ζ), divergence (D), temperature (T)
  and log surface pressure (ln ps).
* Semi-implicit or leapfrog-ready infrastructure with two-thirds dealiasing,
  ∇⁸ hyperdiffusion, vertical diffusion, boundary-layer Rayleigh drag and an
  upper-level sponge.
* Simplified physics: configurable solar heating with either a uniform
  50–80 km vertical profile or a Tomasko (1980) inspired profile, and Newtonian
  cooling following a Crisp-style relaxation timescale.
* I/O helpers for xarray/NetCDF checkpoints and a simple driver that returns a
  Dataset for quick analysis.

The implementation keeps the linear algebra intentionally lightweight so that it
can run quickly on CPU-only GitHub Actions runners while remaining fully JIT
compatible.
