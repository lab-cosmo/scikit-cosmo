import unittest

import numpy as np
from sklearn.datasets import fetch_california_housing as load

from skmatter.sample_selection import CUR, FPS


class TestCUR(unittest.TestCase):
    def setUp(self):
        self.X, _ = load(return_X_y=True)
        self.X = self.X[FPS(n_to_select=100).fit(self.X).selected_idx_]
        self.n_select = min(20, min(self.X.shape) // 2)

    def test_sample_transform(self):
        """
        Check that an error is raised when the transform function is used,
        because sklearn does not support well transformers that change the number
        of samples with other classes like Pipeline
        """
        selector = CUR(n_to_select=1)
        selector.fit(self.X)
        with self.assertRaises(ValueError) as error:
            _ = selector.transform(self.X)

        self.assertTrue(
            "Transform is not currently supported for sample selection."
            == str(error.exception)
        )

    def test_restart(self):
        """Check that the model can be restarted with a new instance"""
        ref_selector = CUR(n_to_select=self.n_select)
        ref_idx = ref_selector.fit(self.X).selected_idx_

        selector = CUR(n_to_select=1)
        selector.fit(self.X)

        for i in range(len(ref_idx) - 2):
            selector.n_to_select += 1
            selector.fit(self.X, warm_start=True)
            self.assertEqual(selector.selected_idx_[i], ref_idx[i])

    def test_non_it(self):
        """Check that the model can be run non-iteratively."""
        K = self.X @ self.X.T
        _, UK = np.linalg.eigh(K)
        ref_idx = np.argsort(-(UK[:, -1] ** 2.0))[: self.n_select]

        selector = CUR(n_to_select=len(ref_idx), recompute_every=0)
        selector.fit(self.X)

        self.assertTrue(np.allclose(selector.selected_idx_, ref_idx))

    def test_unique_selected_idx_zero_score(self):
        """
        Tests that the selected idxs are unique, which may not be the
        case when the score is numerically zero.
        """
        np.random.seed(0)
        n_samples = 10
        n_features = 15
        X = np.random.rand(n_samples, n_features)
        X[4, :] = np.random.rand(15) * 1e-13
        X[5, :] = np.random.rand(15) * 1e-13
        X[6, :] = np.random.rand(15) * 1e-13
        selector_problem = CUR(n_to_select=len(X)).fit(X)
        assert len(selector_problem.selected_idx_) == len(
            set(selector_problem.selected_idx_)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
