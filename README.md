# AFES-Venus JAX (simplified)

This repository provides a lightweight spectral-transform core inspired by AFES-Venus.
The implementation targets educational and testing purposes while keeping the
API JAX-friendly.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running tests

```
pytest -q
```

## Demo

```
python -m afes_venus_jax.examples.t42l60_venus_dry_spinup
```
