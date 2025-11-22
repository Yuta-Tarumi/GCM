import jax.numpy as jnp

import afes_venus_jax.config as cfg
import afes_venus_jax.diffusion as diff


def test_hyperdiffusion_e_fold():
    spec = jnp.zeros((cfg.Lmax + 1, cfg.Lmax + 1), dtype=jnp.complex128)
    ell = cfg.Lmax
    spec = spec.at[ell, 0].set(1.0)
    out = diff.hyperdiffusion(spec, lmax=cfg.Lmax, tau=cfg.tau_hdiff)
    eig = (ell * (ell + 1) / (cfg.a ** 2)) ** cfg.order_hdiff
    eig_max = (cfg.Lmax * (cfg.Lmax + 1) / (cfg.a ** 2)) ** cfg.order_hdiff
    nu = 1.0 / (cfg.tau_hdiff * eig_max)
    expected = spec * jnp.exp(-cfg.dt * nu * eig)
    assert jnp.allclose(out, expected)
