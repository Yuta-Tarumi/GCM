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
