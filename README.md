# AFES Venus JAX (minimal)

A light-weight hydrostatic primitive-equation core for Venus built on top of
[JAX](https://github.com/google/jax). The model follows the specification from
the user story:

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

## Running tests

```bash
pytest -q
```

## Demo

```bash
python -m afes_venus_jax.examples.t42l60_venus_dry_spinup
```

To generate a short demo movie (vorticity at level 0) and write
`t42l60_vort.mp4` to the working directory:

```bash
python -m afes_venus_jax.examples.run_and_movie --scenario vortex_pair
```

Pass `--scenario random_noise` to reproduce the previous near-rest example if
desired. The default `vortex_pair` configuration seeds a balanced pair of
opposite-signed vortices that yields bounded yet visually interesting output for
quick smoke tests.

For a longer superrotation showcase that starts from a noisy state, integrates
for several Venus days, and writes a two-panel animation highlighting the zonal
wind spin-up:

```bash
python -m afes_venus_jax.examples.superrotation_demo --nsteps 4000 --sample-interval 40
```

Adjust `--level-height-km` (default 55 km) to target different layers, and use
`--equator-band` to control how wide of a latitude band feeds the equatorial
mean profile.

To mirror the AFES-Venus paper setup more closely—including the Tomasko-style
solar heating profile, stronger hyperdiffusion, and explicit sponge-layer
damping—use the dedicated configuration and reproduction helper. The defaults
now start from a weaker perturbation (to avoid runaway jets) and the script
prints a quick comparison against the 40–120 m s⁻¹ cloud-top jets reported in
the paper so you can tell if additional spin-up steps are needed:

```bash
python -m afes_venus_jax.examples.afes_paper_reproduction --config afes_venus_jax/config/afes_venus_paper.yaml \
  --nsteps 40 --sample-interval 4 --movie figures/afes_paper_superrotation.gif
```

The script writes a GIF plus diagnostic PNGs under `figures/` that can be
compared against the paper's superrotation experiments.
