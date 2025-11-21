import tempfile
import unittest
from pathlib import Path

import numpy as np

import afes_venus_jax.config as cfg
from afes_venus_jax import grid


class GaussianGridTests(unittest.TestCase):
    def test_t42_grid_matches_leggauss(self):
        lats, lons, weights = grid.gaussian_grid()

        mu, w_ref = np.polynomial.legendre.leggauss(cfg._default_nlat)
        lats_ref = np.arcsin(mu)
        lons_ref = np.linspace(0, 2 * np.pi, cfg._default_nlon, endpoint=False)

        np.testing.assert_equal(lats.shape, (cfg._default_nlat,))
        np.testing.assert_equal(lons.shape, (cfg._default_nlon,))
        np.testing.assert_equal(weights.shape, (cfg._default_nlat,))

        np.testing.assert_allclose(lats, lats_ref)
        np.testing.assert_allclose(lons, lons_ref)
        np.testing.assert_allclose(weights, w_ref)

    def test_grid_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "grid_cache.npz"

            lats_a, lons_a, weights_a = grid.gaussian_grid(
                cache=True, cache_path=cache_path
            )

            self.assertTrue(cache_path.exists())

            lats_b, lons_b, weights_b = grid.gaussian_grid(
                cache=True, cache_path=cache_path
            )

            np.testing.assert_array_equal(lats_a, lats_b)
            np.testing.assert_array_equal(lons_a, lons_b)
            np.testing.assert_array_equal(weights_a, weights_b)


if __name__ == "__main__":
    unittest.main()
