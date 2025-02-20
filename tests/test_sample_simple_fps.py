import unittest

import numpy as np
from sklearn.datasets import load_diabetes as get_dataset
from sklearn.utils.validation import NotFittedError

from skmatter.sample_selection import FPS


class TestFPS(unittest.TestCase):
    def setUp(self):
        self.X, _ = get_dataset(return_X_y=True)
        self.idx = [0, 123, 441, 187, 117, 276, 261, 281, 251, 193]

    def test_restart(self):
        """Checks that the model can be restarted with a new number of samples and
        `warm_start`.
        """
        selector = FPS(n_to_select=1, initialize=self.idx[0])
        selector.fit(self.X)

        for i in range(2, len(self.idx)):
            selector.n_to_select = i
            selector.fit(self.X, warm_start=True)
            self.assertEqual(selector.selected_idx_[i - 1], self.idx[i - 1])

    def test_initialize(self):
        """Checks that the model can be initialized in all applicable manners and throws
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
        """Checks that the hausdorff distances are returnable after fitting."""
        selector = FPS(n_to_select=1)
        selector.fit(self.X)
        _ = selector.get_select_distance()

        with self.assertRaises(NotFittedError):
            selector = FPS(n_to_select=1)
            _ = selector.get_select_distance()

    def test_threshold(self):
        selector = FPS(
            n_to_select=10,
            score_threshold=5e-2,
            score_threshold_type="absolute",
        )
        selector.fit(self.X)
        self.assertEqual(len(selector.selected_idx_), 6)
        self.assertEqual(selector.selected_idx_.tolist(), self.idx[:6])

        selector = FPS(
            n_to_select=10,
            score_threshold=0.4,
            score_threshold_type="relative",
        )
        selector.fit(self.X)
        self.assertEqual(len(selector.selected_idx_), 5)
        self.assertEqual(selector.selected_idx_.tolist(), self.idx[:5])

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
        selector_problem = FPS(n_to_select=len(X)).fit(X)
        assert len(selector_problem.selected_idx_) == len(
            set(selector_problem.selected_idx_)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
