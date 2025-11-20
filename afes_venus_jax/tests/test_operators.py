import jax
import jax.numpy as jnp
from afes_venus_jax.spharm import psi_chi_from_vort_div, uv_from_psi_chi, vort_div_from_uv
from afes_venus_jax.config import DEFAULT_CFG


def test_vort_div_consistency():
    cfg = DEFAULT_CFG
    key = jax.random.PRNGKey(0)
    zeta = jax.random.normal(key, (cfg.nlat, cfg.nlon), dtype=jnp.float64)
    div = jax.random.normal(key, (cfg.nlat, cfg.nlon), dtype=jnp.float64)
    psi, chi = psi_chi_from_vort_div(jnp.fft.fft2(zeta), jnp.fft.fft2(div), cfg)
    u, v = uv_from_psi_chi(psi, chi, cfg)
    vort_back, div_back = vort_div_from_uv(u, v, cfg)
    rel_vort = jnp.linalg.norm(vort_back - jnp.fft.fft2(zeta)) / jnp.linalg.norm(jnp.fft.fft2(zeta))
    rel_div = jnp.linalg.norm(div_back - jnp.fft.fft2(div)) / jnp.linalg.norm(jnp.fft.fft2(div))
    assert rel_vort < 1e7
    assert rel_div < 1e7
