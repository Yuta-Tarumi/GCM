import jax
import jax.numpy as jnp

from afes_venus_jax.config import ModelConfig
from afes_venus_jax.spharm import (
    spectral_shape,
    psi_chi_from_zeta_div,
    uv_from_psi_chi,
    lap_spec,
)


def test_psi_chi_uv_roundtrip():
    cfg = ModelConfig(Lmax=6, nlat=16, nlon=32, L=4)
    key = jax.random.PRNGKey(1)
    zeta = jax.random.normal(key, (cfg.L, *spectral_shape(cfg))) * 1e-6
    div = jax.random.normal(key, (cfg.L, *spectral_shape(cfg))) * 1e-6
    psi, chi = psi_chi_from_zeta_div(zeta, div, cfg)
    u, v = uv_from_psi_chi(psi, chi, cfg)
    # recompute zeta/div from psi/chi via Laplacian
    zeta_back = lap_spec(psi, cfg)
    div_back = lap_spec(chi, cfg)
    err_zeta = jnp.max(jnp.abs(zeta_back - zeta))
    err_div = jnp.max(jnp.abs(div_back - div))
    assert err_zeta < 1e-5
    assert err_div < 1e-5
