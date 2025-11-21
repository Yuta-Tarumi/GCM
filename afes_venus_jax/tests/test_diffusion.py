import jax.numpy as jnp

from afes_venus_jax.config import ModelConfig
from afes_venus_jax.diffusion import hyperdiffuse
from afes_venus_jax.spharm import spectral_shape


def test_hyperdiffusion_decay():
    cfg = ModelConfig(Lmax=6, nlat=16, nlon=32, L=4)
    spec = jnp.zeros(spectral_shape(cfg), dtype=jnp.complex128)
    spec = spec.at[cfg.Lmax, cfg.Lmax + cfg.Lmax].set(1.0)
    spec_new = hyperdiffuse(spec, cfg)
    l = cfg.Lmax
    eig_max = (cfg.Lmax * (cfg.Lmax + 1) / (cfg.a ** 2)) ** cfg.order_hdiff
    nu = 1.0 / (cfg.tau_hdiff * eig_max)
    eig = (l * (l + 1) / (cfg.a ** 2)) ** cfg.order_hdiff
    expected = 1.0 - cfg.dt * nu * eig
    assert jnp.isclose(spec_new[cfg.Lmax, cfg.Lmax + cfg.Lmax].real, expected, rtol=1e-6)
