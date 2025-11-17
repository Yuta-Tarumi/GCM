# AFES Venus JAX (minimal)

Minimal illustrative hydrostatic primitive core built with JAX. The implementation
uses simple pseudo-spectral placeholders to keep tests lightweight while
maintaining JIT-friendly structure.

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
python -m afes_venus_jax.examples.run_and_movie
```
