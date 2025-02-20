import unittest

import numpy as np
from sklearn.datasets import load_diabetes as get_dataset
from sklearn.utils.validation import NotFittedError

from skmatter.feature_selection import FPS


class TestFPS(unittest.TestCase):
    def setUp(self):
        self.X, _ = get_dataset(return_X_y=True)
        self.idx = [0, 6, 1, 2, 4, 9, 3]

    def test_restart(self):
        """
        Check that the model can be restarted with a new number of
        features and `warm_start`
        """
        selector = FPS(n_to_select=1, initialize=self.idx[0])
        selector.fit(self.X)

        for i in range(2, len(self.idx)):
            selector.n_to_select = i
            selector.fit(self.X, warm_start=True)
            self.assertEqual(selector.selected_idx_[i - 1], self.idx[i - 1])

    def test_initialize(self):
        """Check that the model can be initialized in all applicable manners and throws
        an error otherwise.
        """
        for initialize in [self.idx[0], "random"]:
            with self.subTest(initialize=initialize):
                selector = FPS(n_to_select=1, initialize=initialize)
                selector.fit(self.X)

        initialize = self.idx[:4]
        with self.subTest(initialize=initialize):
            selector = FPS(n_to_select=len(self.idx) - 1, initialize=initialize)
            selector.fit(self.X)
            for i in range(4):
                self.assertEqual(selector.selected_idx_[i], self.idx[i])

        initialize = np.array(self.idx[:4])
        with self.subTest(initialize=initialize):
            selector = FPS(n_to_select=len(self.idx) - 1, initialize=initialize)
            selector.fit(self.X)
            for i in range(4):
                self.assertEqual(selector.selected_idx_[i], self.idx[i])

        initialize = np.array([1, 5, 3, 0.25])
        with self.subTest(initialize=initialize):
            with self.assertRaises(ValueError) as cm:
                selector = FPS(n_to_select=len(self.idx) - 1, initialize=initialize)
                selector.fit(self.X)
            self.assertEqual(
                str(cm.exception), "Invalid value of the initialize parameter"
            )

        initialize = np.array([[1, 5, 3], [2, 4, 6]])
        with self.subTest(initialize=initialize):
            with self.assertRaises(ValueError) as cm:
                selector = FPS(n_to_select=len(self.idx) - 1, initialize=initialize)
                selector.fit(self.X)
            self.assertEqual(
                str(cm.exception), "Invalid value of the initialize parameter"
            )

        with self.assertRaises(ValueError) as cm:
            selector = FPS(n_to_select=1, initialize="bad")
            selector.fit(self.X)
        self.assertEqual(str(cm.exception), "Invalid value of the initialize parameter")

    def test_get_distances(self):
        """Check that the hausdorff distances are returnable after fitting."""
        selector = FPS(n_to_select=7)
        selector.fit(self.X)
        d = selector.get_select_distance()

        dist_grad = d[1:-1] - d[2:]
        self.assertTrue(all(dist_grad > 0))

        with self.assertRaises(NotFittedError):
            selector = FPS(n_to_select=7)
            _ = selector.get_select_distance()

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
        selector_problem = FPS(n_to_select=len(X.T)).fit(X)
        assert len(selector_problem.selected_idx_) == len(
            set(selector_problem.selected_idx_)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
