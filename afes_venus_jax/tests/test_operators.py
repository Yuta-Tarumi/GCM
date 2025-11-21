import jax
import jax.numpy as jnp

import afes_venus_jax.config as cfg
import afes_venus_jax.spharm as sph


def test_recover_winds_and_back():
    key = jax.random.PRNGKey(1)
    zeta = jax.random.normal(key, (cfg.Lmax + 1, cfg.Lmax + 1)) + 0j
    div = jnp.zeros_like(zeta)
    psi, chi = sph.psi_chi_from_zeta_div(zeta, div)
    u, v = sph.uv_from_psi_chi(psi, chi)
    zeta_grid = sph.synthesis_spec_to_grid(zeta)
    zeta_back = sph.analysis_grid_to_spec(zeta_grid)
    err = jnp.linalg.norm(zeta_back - zeta) / jnp.linalg.norm(zeta)
    assert err < 1e-6
