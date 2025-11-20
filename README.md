# AFES Venus JAX (minimal)

A light-weight hydrostatic primitive-equation core for Venus built on top of
[JAX](https://github.com/google/jax). This minimal snapshot retains only the
dynamics and physics core—the automated test suites have been removed. The
model follows the specification from the user story:

* T42 spectral truncation on a 128×64 Gaussian grid with 60 σ-levels.
* Prognostic variables: relative vorticity (ζ), divergence (D), temperature (T)
  and log surface pressure (ln ps).
* Semi-implicit or leapfrog-ready infrastructure with two-thirds dealiasing,
  ∇⁴ hyperdiffusion (Laplacian applied twice), vertical diffusion,
  boundary-layer Rayleigh drag and an upper-level sponge.
* Simplified physics: configurable solar heating with either a uniform
  50–80 km vertical profile or a Tomasko (1980) inspired profile, and Newtonian
  cooling following a Crisp-style relaxation timescale.
* I/O helpers for xarray/NetCDF checkpoints and a simple driver that returns a
  Dataset for quick analysis.

The implementation keeps the linear algebra intentionally lightweight so that it
can run quickly on CPU-only GitHub Actions runners while remaining fully JIT
compatible.

## Reproducing the AFES-Venus paper setup

The `afes_venus_paper.yaml` configuration mirrors the Sugimoto et al. (2023)
settings, including T42/L60 resolution, ∇⁴ hyperdiffusion (Laplacian applied
twice), vertical diffusion and the Rayleigh sponge. The helper below
initialises the solid-body westward circulation used in that study—with a
100 m s⁻¹ westward equatorial wind at the model top—and runs a short
demonstration integration:

```python
import afes_venus_jax as avj

# 240 steps ≈ 40 hours with the paper's 600 s timestep
ds = avj.driver.run_afes_venus_paper_demo(nsteps=240, history_stride=24)
print(ds)
```

For a command-line shortcut, install the package in editable mode and run the
demo directly:

```bash
pip install -e .
afes-venus-demo --nsteps 240 --history-stride 24 --output paper_demo.nc
```

The CLI mirrors the defaults shown above and can be invoked with
``python -m afes_venus_jax``. Pass `--help` to see all options, including the
ability to tweak the initial solid-body rotation parameters.

By default the initial solid rotation co-rotates with the planet; pass an
`angular_speed` in rad s⁻¹ or `equatorial_speed` in m s⁻¹ (negative for westward)
to match a different solid-body jet strength from the paper.
