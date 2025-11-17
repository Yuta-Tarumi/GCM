import jax
import jax.numpy as jnp
from afes_venus_jax import diffusion, state, config


def test_diffusion_damps():
    s = state.empty_state(L=1, nlat=config.nlat, nlon=config.nlon)
    mode = jnp.exp(1j * jnp.linspace(0, 2 * jnp.pi, config.nlon))
    s = state.ModelState(
        zeta=jnp.tile(mode, (1, config.nlat, 1)),
        div=jnp.tile(mode, (1, config.nlat, 1)),
        T=jnp.tile(mode, (1, config.nlat, 1)),
        lnps=jnp.tile(mode, (config.nlat, 1)),
    )
    damped = diffusion.apply_all(s, tau=10.0, order=4)
    assert jnp.all(jnp.isfinite(damped.zeta))
