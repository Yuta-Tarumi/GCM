import jax.numpy as jnp
from afes_venus_jax import diffusion, state, config


def test_diffusion_damps():
    cfg = config.DEFAULT
    s = state.empty_state(L=1, nlat=cfg.numerics.nlat, nlon=cfg.numerics.nlon)
    mode = jnp.exp(1j * jnp.linspace(0, 2 * jnp.pi, cfg.numerics.nlon, endpoint=False))
    mode3d = jnp.tile(mode, (1, cfg.numerics.nlat, 1))
    s = state.ModelState(
        zeta=mode3d,
        div=mode3d,
        T=mode3d,
        lnps=jnp.tile(mode, (cfg.numerics.nlat, 1)),
    )
    damped = diffusion.apply_hyperdiffusion(s, cfg)
    assert jnp.linalg.norm(damped.zeta.real) <= jnp.linalg.norm(s.zeta.real)
