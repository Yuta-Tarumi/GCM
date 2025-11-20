import jax
import jax.numpy as jnp
from afes_venus_jax.spharm import psi_chi_from_vort_div, uv_from_psi_chi, vort_div_from_uv
from afes_venus_jax.config import DEFAULT_CFG


def test_vort_div_consistency():
    cfg = DEFAULT_CFG
    zeta = jnp.zeros((cfg.Lmax + 1, cfg.Lmax + 1), dtype=jnp.complex128).at[3, 2].set(1.0)
    div = jnp.zeros_like(zeta)
    psi, chi = psi_chi_from_vort_div(zeta, div, cfg)
    u, v = uv_from_psi_chi(psi, chi, cfg)
    vort_back, div_back = vort_div_from_uv(u, v, cfg)
    rel_vort = jnp.linalg.norm(vort_back - zeta) / jnp.linalg.norm(zeta)
    assert rel_vort < 5e-3
    assert jnp.linalg.norm(div_back) < 5e-3
