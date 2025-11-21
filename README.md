# AFES Venus JAX (simplified)

This repository contains a minimal JAX-based hydrostatic core inspired by the
AFES-Venus configuration. The implementation uses lightweight FFT-based
transforms to keep dependencies small while offering the same API surface as a
full spectral transform model. All arrays run with double precision (complex128
for spectral coefficients) and are JIT compatible.

## Installation

Create a Python environment with Python 3.10+ and install the requirements:

```bash
pip install -r requirements.txt
```

## Running the Venus demo

```bash
python -m afes_venus_jax.examples.t42l60_venus_dry_spinup
```

The script performs a short dry spin-up at T42L60 resolution and prints simple
diagnostics.

## Tests

Run the unit tests with

```bash
pytest
```

## Diurnal forcing notes

* The shortwave heating includes a day–night modulation: `solar_heating` is
  scaled by `max(0, cos(lat) * cos(lon - subsolar_longitude))`, so the
  nightside receives zero shortwave input when
  `solar_diurnal_contrast=1.0`. 【F:afes_venus_jax/tendencies.py†L63-L80】
* Heating is centered near `sigma≈0.02`, which corresponds to the 50–80 km
  cloud deck emphasized by Tomasko et al. (1980) when mapped through the
  exponential sigma grid (`z = -H ln sigma` with `H=15 km`). At the default
  50-day Newtonian cooling timescale, a 240-step (∼1.7 day) run produces only a
  few kelvin of zonal contrast, so fields may look horizontally smooth at early
  times. 【F:afes_venus_jax/config.py†L40-L78】【F:afes_venus_jax/vertical.py†L16-L46】
* To see a sharper day–night signal quickly, try increasing
  `--solar-diurnal-contrast` above the default, raising
  `solar_heating_rate`, or shortening `tau_newtonian`. You can also manually
  sweep `--subsolar-longitude-deg` between runs to mimic a migrating subsolar
  point.
