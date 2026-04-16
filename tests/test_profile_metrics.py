import unittest

import numpy as np

from utils.profile_metrics import (
    compute_profile_sharpness,
    compute_semantic_band,
    interpolate_profile,
    linear_cka,
    maybe_global_layer,
    safe_pearsonr,
    top_k_layers,
)


class ProfileMetricsTest(unittest.TestCase):
    def test_top_k_layers_orders_descending(self):
        layers = [0, 1, 2, 3]
        scores = [0.1, 0.8, 0.3, 0.7]
        self.assertEqual(top_k_layers(layers, scores, k=2), [1, 3])

    def test_profile_sharpness_prefers_clear_peak(self):
        scores = [0.1, 0.2, 1.0, 0.3]
        sharpness = compute_profile_sharpness(scores, 2)
        self.assertGreater(sharpness, 2.0)

    def test_semantic_band_tracks_supported_window(self):
        band = compute_semantic_band([5, 5, 6, 7, 11], band_width=3)
        self.assertEqual(band["band_layers"], [5, 6, 7])
        self.assertEqual(band["support_count"], 4)

    def test_maybe_global_layer_requires_support(self):
        self.assertEqual(maybe_global_layer([4, 4, 4, 5], min_fraction=0.6), 4)
        self.assertIsNone(maybe_global_layer([4, 5, 6, 7], min_fraction=0.6))

    def test_interpolation_preserves_endpoints(self):
        interpolated = interpolate_profile([0, 2, 4], [0.0, 1.0, 0.0], num_points=5)
        self.assertAlmostEqual(float(interpolated[0]), 0.0)
        self.assertAlmostEqual(float(interpolated[2]), 1.0)
        self.assertAlmostEqual(float(interpolated[-1]), 0.0)

    def test_safe_pearson_handles_constant_profiles(self):
        self.assertEqual(safe_pearsonr([1, 1, 1], [0, 1, 2]), 0.0)

    def test_linear_cka_identical_representations(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        score = linear_cka(X, X)
        self.assertAlmostEqual(score, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
