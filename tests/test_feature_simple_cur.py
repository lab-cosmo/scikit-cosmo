import unittest

import numpy as np
from sklearn import exceptions

from skmatter.datasets import load_csd_1000r as load
from skmatter.feature_selection import CUR, FPS


class TestCUR(unittest.TestCase):
    def setUp(self):
        self.X, _ = load(return_X_y=True)
        self.X = FPS(n_to_select=10).fit(self.X).transform(self.X)

    def test_bad_transform(self):
        selector = CUR(n_to_select=2)
        with self.assertRaises(exceptions.NotFittedError):
            _ = selector.transform(self.X)

    def test_restart(self):
        """Check that the model can be restarted with a new instance."""
        ref_selector = CUR(n_to_select=self.X.shape[-1] - 3).fit(X=self.X)
        ref_idx = ref_selector.selected_idx_

        selector = CUR(n_to_select=1)
        selector.fit(self.X)

        for i in range(self.X.shape[-1] - 3):
            selector.n_to_select += 1
            selector.fit(self.X, warm_start=True)
            self.assertEqual(selector.selected_idx_[i], ref_idx[i])

    def test_non_it(self):
        """Check that the model can be run non-iteratively."""
        C = self.X.T @ self.X
        _, UC = np.linalg.eigh(C)
        ref_idx = np.argsort(-(UC[:, -1] ** 2.0))[:-1]

        selector = CUR(n_to_select=self.X.shape[-1] - 1, recompute_every=0)
        selector.fit(self.X)

        self.assertTrue(np.allclose(selector.selected_idx_, ref_idx))

    def test_unique_selected_idx_zero_score(self):
        """
        Tests that the selected idxs are unique, which may not be the
        case when the score is numerically zero
        """
        np.random.seed(0)
        n_samples = 10
        n_features = 15
        X = np.random.rand(n_samples, n_features)
        X[:, 3] = np.random.rand(10) * 1e-13
        X[:, 4] = np.random.rand(10) * 1e-13
        selector_problem = CUR(n_to_select=len(X.T)).fit(X)
        assert len(selector_problem.selected_idx_) == len(
            set(selector_problem.selected_idx_)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
