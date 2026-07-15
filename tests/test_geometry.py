import unittest

import numpy as np

from experiments.adequacy import (
    leave_one_out_adequacy_margins,
    normalized_adequacy_margin,
)
from experiments.gdv_experiments import (
    compute_gdv,
    compute_mean_inter_class_distance,
    compute_mean_intra_class_distance,
)


class NormalizedMarginTests(unittest.TestCase):
    def test_centroid_endpoints_are_bounded(self):
        c0 = np.array([0.0, 0.0])
        c1 = np.array([3.0, 4.0])
        self.assertAlmostEqual(normalized_adequacy_margin(c0, c0, c1), 1.0)
        self.assertAlmostEqual(normalized_adequacy_margin(c1, c0, c1), -1.0)

    def test_random_points_obey_reverse_triangle_bound(self):
        rng = np.random.default_rng(7)
        c0, c1 = rng.normal(size=(2, 19))
        values = [
            normalized_adequacy_margin(h, c0, c1)
            for h in rng.normal(size=(200, 19))
        ]
        self.assertLessEqual(max(abs(value) for value in values), 1.0 + 1e-12)

    def test_leave_one_out_removes_self_centroid_advantage(self):
        X = np.array([[0.0], [10.0], [4.0], [6.0]])
        labels = np.array([0, 0, 1, 1])
        raw, normalized = leave_one_out_adequacy_margins(X, labels)
        self.assertTrue(np.all(raw < 0.0))
        self.assertTrue(np.all(np.abs(normalized) <= 1.0))


class GdvTests(unittest.TestCase):
    def test_matches_published_mean_distance_equation(self):
        X = np.array([[0.0], [1.0], [4.0], [5.0]])
        labels = np.array([0, 0, 1, 1])
        scaled = 0.5 * (X - X.mean(axis=0)) / X.std(axis=0)
        expected = (
            compute_mean_intra_class_distance(scaled, labels)
            - compute_mean_inter_class_distance(scaled, labels)
        ) / np.sqrt(X.shape[1])
        self.assertAlmostEqual(compute_gdv(X, labels), expected)

    def test_translation_and_feature_scaling_invariance(self):
        X = np.array([[0.0, 2.0], [1.0, 4.0], [4.0, 8.0], [5.0, 10.0]])
        labels = np.array([0, 0, 1, 1])
        transformed = X * np.array([7.0, 0.25]) + np.array([91.0, -13.0])
        self.assertAlmostEqual(compute_gdv(X, labels), compute_gdv(transformed, labels))


if __name__ == "__main__":
    unittest.main()
