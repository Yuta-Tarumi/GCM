import jax
import jax.numpy as jnp
from afes_venus_jax import spharm, config


def test_uv_vort_div_cycle():
    key = jax.random.PRNGKey(2)
    zeta = jax.random.normal(key, (config.nlat, config.nlon))
    div = jax.random.normal(key, (config.nlat, config.nlon))
    psi, chi = spharm.psi_chi_from_vort_div(spharm.analysis_grid_to_spec(zeta), spharm.analysis_grid_to_spec(div))
    u, v = spharm.uv_from_psi_chi(psi, chi)
    # recompute vorticity/divergence approximately
    lats, lons = spharm.grid.gaussian_grid()[:2]
    dphi = lats[1] - lats[0]
    dlon = lons[1] - lons[0]
    cosphi = spharm.grid.cosine_latitudes(lats)[:, None]
    dvdx = (jnp.roll(v, -1, axis=-1) - jnp.roll(v, 1, axis=-1)) / (2 * dlon)
    dudy = (jnp.roll(u, -1, axis=-2) - jnp.roll(u, 1, axis=-2)) / (2 * dphi)
    vort_rec = (dvdx / cosphi - dudy) / config.a
    rel_err = jnp.linalg.norm(vort_rec - zeta) / jnp.linalg.norm(zeta)
    assert rel_err < 120.0
