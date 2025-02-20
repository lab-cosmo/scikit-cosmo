"""
Data sub-selection modules primarily corresponding to methods derived from CUR matrix
decomposition and Farthest Point Sampling. In their classical form, CUR and FPS
determine a data subset that maximizes the variance (CUR) or distribution (FPS) of the
features or samples. These methods can be modified to combine supervised target
information denoted by the methods `PCov-CUR` and `PCov-FPS`. For further reading, refer
to [Imbalzano2018]_ and [Cersonsky2021]_. These selectors can be used for both feature
and sample selection, with similar instantiations. All sub-selection methods  scores
each feature or sample (without an estimator) and chooses that with the maximum score. A
simple example of usage:

.. doctest::

    >>> # feature selection
    >>> import numpy as np
    >>> from skmatter.feature_selection import CUR, FPS, PCovCUR, PCovFPS
    >>> selector = CUR(
    ...     # the number of selections to make
    ...     # if None, set to half the samples or features
    ...     # if float, fraction of the total dataset to select
    ...     # if int, absolute number of selections to make
    ...     n_to_select=2,
    ...     # option to use `tqdm <https://tqdm.github.io/>`_ progress bar
    ...     progress_bar=True,
    ...     # float, cutoff score to stop selecting
    ...     score_threshold=1e-12,
    ...     # boolean, whether to select randomly after non-redundant selections
    ...     # are exhausted
    ...     full=False,
    ... )
    >>> X = np.array(
    ...     [
    ...         [0.12, 0.21, 0.02],  # 3 samples, 3 features
    ...         [-0.09, 0.32, -0.10],
    ...         [-0.03, -0.53, 0.08],
    ...     ]
    ... )
    >>> y = np.array([0.0, 0.0, 1.0])  # classes of each sample
    >>> selector.fit(X)
    CUR(n_to_select=2, progress_bar=True, score_threshold=1e-12)
    >>> Xr = selector.transform(X)
    >>> print(Xr.shape)
    (3, 2)
    >>> selector = PCovCUR(n_to_select=2)
    >>> selector.fit(X, y)
    PCovCUR(n_to_select=2)
    >>> Xr = selector.transform(X)
    >>> print(Xr.shape)
    (3, 2)
    >>>
    >>> # Now sample selection
    >>> from skmatter.sample_selection import CUR, FPS, PCovCUR, PCovFPS
    >>> selector = CUR(n_to_select=2)
    >>> selector.fit(X)
    CUR(n_to_select=2)
    >>> Xr = X[selector.selected_idx_]
    >>> print(Xr.shape)
    (2, 3)

These selectors are available:

* :ref:`CUR-api`: a decomposition: an iterative feature selection method based upon the
  singular value decoposition.
* :ref:`PCov-CUR-api` decomposition extends upon CUR by using augmented right or left
  singular vectors inspired by Principal Covariates Regression.
* :ref:`FPS-api`: a common selection technique intended to exploit the diversity of the
  input space. The selection of the first point is made at random or by a separate
  metric
* :ref:`PCov-FPS-api` extends upon FPS much like PCov-CUR does to CUR.
* :ref:`Voronoi-FPS-api`: conduct FPS selection, taking advantage of Voronoi
  tessellations to accelerate selection.
* :ref:`DCH-api`: selects samples by constructing a directional convex hull and
  determining which samples lie on the bounding surface.
"""

import numbers
import warnings
from abc import abstractmethod

import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils import check_array, check_random_state, check_X_y, safe_mask
from sklearn.utils.validation import FLOAT_DTYPES, as_float_array, check_is_fitted

from .utils import (
    X_orthogonalizer,
    Y_feature_orthogonalizer,
    Y_sample_orthogonalizer,
    get_progress_bar,
    no_progress_bar,
    pcovr_covariance,
    pcovr_kernel,
)


class GreedySelector(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    """Transformer that adds, via greedy forward selection,
    features or samples to form a subset. At each stage, the model scores each
    feature or sample (without an estimator) and chooses that with the maximum score.

    Parameters
    ----------
    selection_type : str, {'feature', 'sample'}
        whether to choose a subset of columns ('feature') or rows ('sample').
        Stored in :py:attr:`self._axis_name` (as text) and :py:attr:`self._axis`
        (as 0 or 1 for 'sample' or 'feature', respectively).
    n_to_select : int or float, default=None
        The number of selections to make. If `None`, half of the features or samples are
        selected. If integer, the parameter is the absolute number of selections
        to make. If float between 0 and 1, it is the fraction of the total dataset to
        select. Stored in :py:attr:`self.n_to_select`.
    score_threshold : float, default=None
        Threshold for the score. If `None` selection will continue until the
        n_to_select is chosen. Otherwise will stop when the score falls below the
        threshold. Stored in :py:attr:`self.score_threshold`.
    score_threshold_type : str, default="absolute"
        How to interpret the ``score_threshold``. When "absolute", the score used by
        the selector is compared to the threshold directly. When "relative", at each
        iteration, the score used by the selector is compared proportionally to the
        score of the first selection, i.e. the selector quits when ``current_score /
        first_score < threshold``. Stored in :py:attr:`self.score_threshold_type`.
    progress_bar: bool, default=False
              option to use `tqdm <https://tqdm.github.io/>`_
              progress bar to monitor selections. Stored in
              :py:attr:`self.report_progress_`.
    full : bool, default=False
        In the case that all non-redundant selections are exhausted, choose
        randomly from the remaining features. Stored in :py:attr:`self.full`.
    random_state : int or :class:`numpy.random`RandomState` instance, default=0

    Attributes
    ----------
    n_selected_ : int
        Counter tracking the number of selections that have been made
    X_selected_ : numpy.ndarray,
        Matrix containing the selected samples or features, for use in fitting
    y_selected_ : numpy.ndarray,
        In sample selection, the matrix containing the selected targets, for use in
        fitting
    """

    def __init__(
        self,
        selection_type,
        n_to_select=None,
        score_threshold=None,
        score_threshold_type="absolute",
        progress_bar=False,
        full=False,
        random_state=0,
    ):
        self.selection_type = selection_type
        self.n_to_select = n_to_select
        self.score_threshold = score_threshold
        self.score_threshold_type = score_threshold_type
        if self.score_threshold_type not in ["relative", "absolute"]:
            raise ValueError(
                "invalid score_threshold_type, expected one of 'relative' or 'absolute'"
            )

        self.full = full
        self.progress_bar = progress_bar
        self.random_state = random_state

    def fit(self, X, y=None, warm_start=False):
        """Learn the features to select.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training vectors.
        y : numpy.ndarray of shape (n_samples,), default=None
            Target values.
        warm_start : bool
            Whether the fit should continue after having already
            run, after increasing n_to_select.
            Assumes it is called with the same X and y

        Returns
        -------
        self : object
        """
        if self.selection_type == "feature":
            self._axis = 1
        elif self.selection_type == "sample":
            self._axis = 0
        else:
            raise ValueError("Only feature and sample selection supported.")

        if self.full and self.score_threshold is not None:
            raise ValueError(
                "You cannot specify both `score_threshold` and `full=True`."
            )

        if self.progress_bar is True:
            self.report_progress_ = get_progress_bar()
        elif self.progress_bar is False:
            self.report_progress_ = no_progress_bar

        params = dict(ensure_min_samples=2, ensure_min_features=2, dtype=FLOAT_DTYPES)

        if hasattr(self, "mixing") or y is not None:
            X, y = self._validate_data(X, y, **params)
            X, y = check_X_y(X, y, multi_output=True)

            if len(y.shape) == 1:
                # force y to have multi_output 2D format even when it's 1D, since
                # many functions, most notably PCov routines, assume an array storage
                # format, most notably to compute (y @ y.T)
                y = y.reshape((len(y), 1))

        else:
            X = check_array(X, **params)

        if self.full and self.score_threshold is not None:
            raise ValueError(
                "You cannot specify both `score_threshold` and `full=True`."
            )

        n_to_select_from = X.shape[self._axis]
        self.n_samples_in_, self.n_features_in_ = X.shape

        error_msg = (
            "n_to_select must be either None, an "
            f"integer in [1, n_{self.selection_type}s] "
            "representing the absolute "
            f"number of {self.selection_type}s, or a float in (0, 1] "
            f"representing a percentage of {self.selection_type}s to "
            f"select. Got {self.n_to_select} {self.selection_type}s and "
            f"an input with {n_to_select_from} {self.selection_type}."
        )

        if self.n_to_select is None:
            n_iterations = n_to_select_from // 2
        elif isinstance(self.n_to_select, numbers.Integral):
            if not 0 < self.n_to_select <= n_to_select_from:
                raise ValueError(error_msg)
            n_iterations = self.n_to_select
        elif isinstance(self.n_to_select, numbers.Real):
            if not 0 < self.n_to_select <= 1:
                raise ValueError(error_msg)
            n_iterations = int(n_to_select_from * self.n_to_select)
        else:
            raise ValueError(error_msg)

        if warm_start:
            if not hasattr(self, "n_selected_") or self.n_selected_ == 0:
                raise ValueError(
                    "Cannot fit with warm_start=True without having been previously"
                    " initialized."
                )
            self._continue_greedy_search(X, y, n_iterations)
        else:
            self._init_greedy_search(X, y, n_iterations)

        n_iterations -= self.n_selected_

        for n in self.report_progress_(range(n_iterations)):
            new_idx = self._get_best_new_selection(self.score, X, y)
            if new_idx is not None:
                self._update_post_selection(X, y, new_idx)
            else:
                warnings.warn(
                    f"Score threshold of {self.score_threshold} reached."
                    f"Terminating search at {self.n_selected_} / {self.n_to_select}.",
                    stacklevel=1,
                )
                self.X_selected_ = np.take(
                    self.X_selected_, np.arange(self.n_selected_), axis=self._axis
                )

                if hasattr(self, "y_selected_"):
                    self.y_selected_ = self.y_selected_[:n]

                self.selected_idx_ = self.selected_idx_[:n]
                self._postprocess(X, y)
                return self

        self._postprocess(X, y)
        return self

    def transform(self, X, y=None):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : numpy.ndarray of shape [n_samples, n_features]
            The input samples.
        y : ignored

        Returns
        -------
        X_r : numpy.ndarray
            The selected subset of the input.
        """
        check_is_fitted(self, ["_axis", "selected_idx_", "n_selected_"])

        if self._axis == 0:
            raise ValueError(
                "Transform is not currently supported for sample selection."
            )

        mask = self.get_support()

        X = check_array(X)

        if len(X.shape) == 1:
            if self._axis == 0:
                X = X.reshape(-1, 1)
            else:
                X = X.reshape(1, -1)

        if len(mask) != X.shape[self._axis]:
            raise ValueError(
                "X has a different shape than during fitting. Reshape your data."
            )
        if self._axis == 1:
            return X[:, safe_mask(X, mask)]
        else:
            return X[safe_mask(X, mask)]

    @abstractmethod
    def score(self, X, y):
        """
        A single str or a callable to evaluate the features or samples
        that is overwritten by the subclass.
        It is assumed that the next selection is that which maximizes the score.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.


        Parameters
        ----------
        X : numpy.ndarray of shape [n_samples, n_features]
            The input samples.
        y : ignored

        Returns
        -------
        score : numpy.ndarray of (n_to_select_from_)
            Scores of the given features or samples
        """
        pass

    def get_support(self, indices=False, ordered=False):
        """Get a mask, or integer index, of the subset

        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of integers, rather
            than a bool mask.

        ordered : bool, default=False
            With indices, if True, the return value will be an array of integers, rather
            than a bool mask, in the order in which they were selected.

        Returns
        -------
        support : An index that selects the retained subset from a original vectors.
                  If indices is False, this is a bool array of shape [# input], in which
                  an element is True iff its corresponding feature or sample is selected
                  for retention. If indices is True, this is an integer array of shape
                  [# n_to_select] whose values are indices into the input vectors.

        """
        check_is_fitted(self, ["support_", "selected_idx_"])
        if indices:
            if ordered:
                return self.selected_idx_
            else:
                return list(sorted(self.selected_idx_))
        else:
            return self._get_support_mask()

    def _init_greedy_search(self, X, y, n_to_select):
        """Initializes the search. Prepares an array to store the selected features."""
        self.n_selected_ = 0
        self.first_score_ = None

        sel_shape = list(X.shape)
        sel_shape[self._axis] = n_to_select

        self.X_selected_ = np.zeros(sel_shape, float)

        if y is not None and self._axis == 0:
            self.y_selected_ = np.zeros(
                (n_to_select, y.reshape(y.shape[0], -1).shape[1]), float
            )
        self.selected_idx_ = np.zeros((n_to_select), int)

    def _continue_greedy_search(self, X, y, n_to_select):
        """Continues the search. Prepares an array to store the selected features."""
        n_pad = [(0, 0), (0, 0)]
        n_pad[self._axis] = (0, n_to_select - self.n_selected_)

        self.X_selected_ = np.pad(
            self.X_selected_,
            n_pad,
            "constant",
            constant_values=0.0,
        )

        if hasattr(self, "y_selected_"):
            self.y_selected_ = np.pad(
                self.y_selected_,
                n_pad,
                "constant",
                constant_values=0.0,
            )

        old_idx = self.selected_idx_.copy()
        self.selected_idx_ = np.zeros((n_to_select), int)
        self.selected_idx_[: self.n_selected_] = old_idx

    def _get_best_new_selection(self, scorer, X, y):
        scores = scorer(X, y)

        # Get the score argmax, but only for idxs not already selected
        _tmp_scores = {
            i: score for i, score in enumerate(scores) if i not in self.selected_idx_
        }
        max_score_idx = max(_tmp_scores, key=_tmp_scores.get)
        if self.score_threshold is not None:
            if self.first_score_ is None:
                self.first_score_ = scores[max_score_idx]

            if self.score_threshold_type == "absolute":
                if scores[max_score_idx] < self.score_threshold:
                    return None

            if self.score_threshold_type == "relative":
                if scores[max_score_idx] / self.first_score_ < self.score_threshold:
                    return None

        return max_score_idx

    def _update_post_selection(self, X, y, last_selected):
        """Saves the most recently selected feature and increments the feature
        counter.
        """
        if self._axis == 1:
            self.X_selected_[:, self.n_selected_] = np.take(
                X, last_selected, axis=self._axis
            )
        else:
            self.X_selected_[self.n_selected_] = np.take(
                X, last_selected, axis=self._axis
            )

            if hasattr(self, "y_selected_"):
                self.y_selected_[self.n_selected_] = y[last_selected]

        self.selected_idx_[self.n_selected_] = last_selected
        self.n_selected_ += 1

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which subset has been selected

        Raises
        ------
        NotFittedError
            If the selector has not yet been fitted

        Returns
        -------
        support : bool or numpy.ndarray of shape [# input]
            An element is True iff its corresponding feature or sample is selected for
            retention.
        """
        check_is_fitted(self, ["support_"])
        return self.support_

    def _postprocess(self, X, y):
        """Post-process X and / or y when selection is finished"""
        self.support_ = np.full(X.shape[self._axis], False)
        self.support_[self.selected_idx_] = True

    def _more_tags(self):
        return {
            "requires_y": False,
        }


class _CUR(GreedySelector):
    """Transformer that performs Greedy Selection by choosing features
    which maximize the magnitude of the right or left singular vectors, consistent with
    classic CUR matrix decomposition.

    .. warning::
        This base class should never be directly instantiated. Instead, use
        :py:class:`skmatter.feature_selection.CUR` and
        :py:class:`skmatter.sample_selection.CUR`, which have the same constructor
        signature.

    Parameters
    ----------
    recompute_every : int
        number of steps after which to recompute the pi score
        defaults to 1, if 0 no re-computation is done
    k : int
        number of eigenvectors to compute the importance score with, defaults to 1
    tolerance: float
        threshold below which scores will be considered 0, defaults to 1E-12

    Attributes
    ----------
    X_current_ : numpy.ndarray (n_samples, n_features)
        The original matrix orthogonalized by previous selections
    """

    def __init__(
        self,
        selection_type,
        recompute_every=1,
        k=1,
        tolerance=1e-12,
        n_to_select=None,
        score_threshold=None,
        score_threshold_type="absolute",
        progress_bar=False,
        full=False,
        random_state=0,
    ):
        self.k = k
        self.tolerance = tolerance
        self.recompute_every = recompute_every

        super().__init__(
            selection_type=selection_type,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            score_threshold_type=score_threshold_type,
            progress_bar=progress_bar,
            full=full,
            random_state=random_state,
        )

    def score(self, X, y=None):
        r"""Returns the importance score of the given samples or features.

        .. note::
            This function does not compute the importance score each time it is called,
            in order to avoid unnecessary computations. This is done by
            :py:func:`self._compute_pi`.

        Parameters
        ----------
        X : numpy.ndarray of shape [n_samples, n_features]
            The input samples.
        y : ignored

        Returns
        -------
        score : numpy.ndarray of (n_to_select_from_)
            :math:`\pi` importance for the given samples or features
        """
        return self.pi_

    def _init_greedy_search(self, X, y, n_to_select):
        """Initializes the search. Prepares an array to store the selected
        features and computes their initial importance score.
        """
        self.X_current_ = as_float_array(X.copy())
        self.pi_ = self._compute_pi(self.X_current_)

        super()._init_greedy_search(X, y, n_to_select)

    def _continue_greedy_search(self, X, y, n_to_select):
        """Continues the search. Prepares an array to store the selected features,
        orthogonalizes the features by those already selected, and computes their
        initial importance.
        """
        for c in self.selected_idx_:
            if self.recompute_every != 0 and (
                np.linalg.norm(np.take(self.X_current_, [c], axis=self._axis))
                > self.tolerance
            ):
                self._orthogonalize(last_selected=c)

        self.pi_ = self._compute_pi(self.X_current_)

        super()._continue_greedy_search(X, y, n_to_select)

    def _compute_pi(self, X, y=None):
        r"""For feature selection, the importance score :math:`\pi` is the sum over
        the squares of the first :math:`k` components of the right singular vectors

        .. math::
            \pi_j = \sum_i^k \left(\mathbf{U}_\mathbf{C}\right)_{ij}^2.

        where :math:`\mathbf{C} = \mathbf{X}^T\mathbf{X}`.

        For sample selection, the importance score :math:`\pi` is the sum over the
        squares of the first :math:`k` components of the right singular vectors

        .. math::
            \pi_j = \sum_i^k \left(\mathbf{U}_\mathbf{K}\right)_{ij}^2.

        where :math:`\mathbf{K} = \mathbf{X}\mathbf{X}^T`.

        Parameters
        ----------
        X : numpy.ndarray of shape [n_samples, n_features]
            The input samples.
        y : ignored

        Returns
        -------
        pi : numpy.ndarray of (n_to_select_from_)
            :math:`\pi` importance for the given samples or features
        """
        svd_kwargs = dict(k=self.k, random_state=self.random_state)
        if self._axis == 0:
            svd_kwargs["return_singular_vectors"] = "u"
            U, _, _ = scipy.sparse.linalg.svds(X, **svd_kwargs)
            U = np.real(U)
            new_pi = (U[:, : self.k] ** 2.0).sum(axis=1)
        else:
            svd_kwargs["return_singular_vectors"] = "vh"
            _, _, Vt = scipy.sparse.linalg.svds(X, **svd_kwargs)
            new_pi = (np.real(Vt) ** 2.0).sum(axis=0)

        return new_pi

    def _update_post_selection(self, X, y, last_selected):
        """
        Saves the most recently selected feature, increments the feature counter,
        and, if the CUR is iterative (if recompute_every>0), orthogonalizes the
        remaining features by the most recently selected.
        """
        super()._update_post_selection(X, y, last_selected)

        if self.recompute_every != 0:
            self._orthogonalize(last_selected)

            if self.n_selected_ % self.recompute_every == 0:
                self.pi_ = self._compute_pi(self.X_current_)

        self.pi_[last_selected] = 0.0

    def _orthogonalize(self, last_selected):
        if self._axis == 1:
            self.X_current_ = X_orthogonalizer(
                x1=self.X_current_, c=last_selected, tol=self.tolerance
            )
        else:
            self.X_current_ = X_orthogonalizer(
                x1=self.X_current_.T, c=last_selected, tol=self.tolerance
            ).T


class _PCovCUR(GreedySelector):
    r"""Transformer that performs Greedy Selection by choosing features
    which maximize the magnitude of the right or left augmented singular vectors.
    This is done by employing the augmented kernel and covariance matrices,

    **WARNING**: This base class should never be directly instantiated.
    Instead, use :py:class:`skmatter.feature_selection.PCovCUR` and
    :py:class:`skmatter.sample_selection.PCovCUR`,
    which have the same constructor signature.

    Parameters
    ----------
    recompute_every : int
        number of steps after which to recompute the pi score defaults to 1, if 0 no
        re-computation is done
    k : int
        number of eigenvectors to compute the importance score with, defaults to 1
    tolerance: float
        threshold below which scores will be considered 0, defaults to 1E-12
    mixing: float, default=0.5
        The PCovR mixing parameter, as described in PCovR as
        :math:`{\alpha}`. Stored in :py:attr:`self.mixing`.

    Attributes
    ----------
    X_current_ : numpy.ndarray (n_samples, n_features)
        The original matrix orthogonalized by previous selections
    y_current_ : numpy.ndarray (n_samples, n_properties)
        The targets orthogonalized by a regression on the previous selections.
    """

    def __init__(
        self,
        selection_type,
        mixing=0.5,
        recompute_every=1,
        k=1,
        tolerance=1e-12,
        n_to_select=None,
        score_threshold=None,
        score_threshold_type="absolute",
        progress_bar=False,
        full=False,
        random_state=0,
    ):
        self.mixing = mixing

        self.k = k
        self.recompute_every = recompute_every
        self.tolerance = tolerance

        super().__init__(
            selection_type=selection_type,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            score_threshold_type=score_threshold_type,
            progress_bar=progress_bar,
            full=full,
            random_state=random_state,
        )

    def score(self, X, y=None):
        r"""Returns the importance score of the given samples or features.

        .. note::
            This function does not compute the importance score each time it is called,
            in order to avoid unnecessary computations. This is done by
            :py:func:`self._compute_pi`.

        Parameters
        ----------
        X : ignored
        y : ignored

        Returns
        -------
        score : numpy.ndarray of (n_to_select_from_)
            :math:`\pi` importance for the given samples or features
        """
        return self.pi_

    def _init_greedy_search(self, X, y, n_to_select):
        """Initializes the search. Prepares an array to store the selected
        features and computes their initial importance score.
        """
        self.X_ref_ = X
        self.y_ref_ = y
        self.X_current_ = X.copy()
        if y is not None:
            self.y_current_ = y.copy()
        else:
            self.y_current_ = None
        self.pi_ = self._compute_pi(self.X_current_, self.y_current_)

        super()._init_greedy_search(X, y, n_to_select)

    def _continue_greedy_search(self, X, y, n_to_select):
        """Continues the search. Prepares an array to store the selected
        features, orthogonalizes the features by those already selected, and computes
        their initial importance.
        """
        for c in self.selected_idx_:
            if self.recompute_every != 0 and (
                np.linalg.norm(np.take(self.X_current_, [c], axis=self._axis))
                > self.tolerance
            ):
                self._orthogonalize(last_selected=c)

        self.pi_ = self._compute_pi(self.X_current_, self.y_current_)

        super()._continue_greedy_search(X, y, n_to_select)

    def _update_post_selection(self, X, y, last_selected):
        """
        Saves the most recently selected feature, increments the feature counter, and,
        if the CUR is iterative (if recompute_every>0), orthogonalizes the remaining
        features by the most recently selected.
        """
        super()._update_post_selection(X, y, last_selected)

        if self.recompute_every != 0:
            self._orthogonalize(last_selected)

            if self.n_selected_ % self.recompute_every == 0:
                self.pi_ = self._compute_pi(self.X_current_, self.y_current_)

        self.pi_[last_selected] = 0.0

    def _compute_pi(self, X, y=None):
        r"""For feature selection, the importance score :math:`\pi` is the sum over
        the squares of the first :math:`k` components of the right singular vectors.

        .. math::
            \pi_j =
            \sum_i^k \left(\mathbf{U}_\mathbf{\tilde{C}}\right)_{ij}^2.

        where :math:`{\mathbf{\tilde{C}} = \alpha \mathbf{X}^T\mathbf{X} +
        (1 - \alpha)(\mathbf{X}^T\mathbf{X})^{-1/2}\mathbf{X}^T
        \mathbf{\hat{Y}\hat{Y}}^T\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1/2}}`
        for some mixing parameter :math:`{\alpha}`. When :math:`{\alpha = 1}`,
        this defaults to the covariance matrix
        :math:`{\mathbf{C} = \mathbf{X}^T\mathbf{X}}` used in CUR.

        For sample selection, the importance score :math:`\pi` is the sum over
        the squares of the first :math:`k` components of the right singular vectors

        .. math::
            \pi_j =
            \sum_i^k \left(\mathbf{U}_\mathbf{\tilde{K}}\right)_{ij}^2.

        where :math:`{\mathbf{\tilde{K}} = \alpha \mathbf{XX}^T +
        (1 - \alpha)\mathbf{\hat{Y}\hat{Y}}^T}` for some mixing parameter
        :math:`{\alpha}`. When :math:`{\alpha = 1}`, this defaults to the Gram
        matrix :math:`{\mathbf{K} = \mathbf{X}\mathbf{X}^T}`.

        Parameters
        ----------
        X : numpy.ndarray of shape [n_samples, n_features]
            The input samples.
        y : ignored

        Returns
        -------
        pi : numpy.ndarray of (n_to_select_from_)
            :math:`\pi` importance for the given samples or features
        """
        if self._axis == 0:
            pcovr_distance = pcovr_kernel(
                self.mixing,
                X,
                y,
            )
        else:
            pcovr_distance = pcovr_covariance(
                self.mixing,
                X,
                y,
                rcond=1e-12,
                rank=None,
            )

        if self.k < pcovr_distance.shape[0] - 1:
            v, U = eigsh(pcovr_distance, k=self.k, tol=1e-12)
        else:
            v, U = eigh(pcovr_distance)
        U = U[:, np.flip(np.argsort(v))]
        pi = (np.real(U)[:, : self.k] ** 2.0).sum(axis=1)

        return pi

    def _orthogonalize(self, last_selected):
        if self._axis == 1:
            self.X_current_ = X_orthogonalizer(
                x1=self.X_current_, c=last_selected, tol=self.tolerance
            )
        else:
            self.X_current_ = X_orthogonalizer(
                x1=self.X_current_.T, c=last_selected, tol=self.tolerance
            ).T
        if self.y_current_ is not None:
            if self._axis == 1:
                self.y_current_ = Y_feature_orthogonalizer(
                    self.y_current_, X=self.X_selected_, tol=self.tolerance
                )
            else:
                self.y_current_ = Y_sample_orthogonalizer(
                    self.y_ref_,
                    self.X_ref_,
                    y_ref=self.y_selected_[: self.n_selected_],
                    X_ref=self.X_selected_[: self.n_selected_],
                    tol=self.tolerance,
                )


class _FPS(GreedySelector):
    """Transformer that performs Greedy Selection using Farthest Point Sampling.

    .. warning::
        This base class should never be directly instantiated. Instead, use
        :py:class:`skmatter.feature_selection.FPS` and
        :py:class:`skmatter.sample_selection.FPS`, which have the same constructor
        signature.

    Parameters
    ----------
    initialize: int, list of int, numpy.ndarray of int, or 'random', default=0
        Index of the first selection(s). If 'random', picks a random
        value when fit starts. Stored in :py:attr:`self.initialize`.


    """

    def __init__(
        self,
        selection_type,
        initialize=0,
        n_to_select=None,
        score_threshold=None,
        score_threshold_type="absolute",
        progress_bar=False,
        full=False,
        random_state=0,
    ):
        self.initialize = initialize

        super().__init__(
            selection_type=selection_type,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            score_threshold_type=score_threshold_type,
            progress_bar=progress_bar,
            full=full,
            random_state=random_state,
        )

    def score(self, X, y=None):
        """
        Returns the Hausdorff distances of all samples to previous selections

        NOTE: This function does not compute the importance score each time it
        is called, in order to avoid unnecessary computations. The hausdorff
        distance is updated in :py:func:`self._update_hausdorff`

        Parameters
        ----------
        X : ignored
        y : ignored

        Returns
        -------
        hausdorff : Hausdorff distances
        """
        return self.hausdorff_

    def get_distance(self):
        r"""Traditional FPS employs a column-wise Euclidean
        distance for feature selection, which can be expressed using the covariance
        matrix :math:`\mathbf{C} = \mathbf{X} ^ T \mathbf{X}`

        .. math::
            \operatorname{d}_c(i, j) = C_{ii} - 2 C_{ij} + C_{jj}.

        For sample selection, this is a row-wise Euclidean distance, which can be
        expressed in terms of the Gram matrix
        :math:`\mathbf{K} = \mathbf{X} \mathbf{X} ^ T`

        .. math::
            \operatorname{d}_r(i, j) = K_{ii} - 2 K_{ij} + K_{jj}.

        Returns
        -------
        hausdorff : numpy.ndarray of shape (`n_to_select_from_`)
            the minimum distance from each point to the set of selected points. once a
            point is selected, the distance is not updated; the final list will reflect
            the distances when selected.
        """
        return self.hausdorff_

    def get_select_distance(self):
        """
        Returns
        -------
        hausdorff_at_select : numpy.ndarray of shape (`n_to_select`)
            at the time of selection, the minimum distance from each selected point to
            the set of previously selected points.
        """
        mask = self.get_support(indices=True, ordered=True)
        return self.hausdorff_at_select_[mask]

    def _init_greedy_search(self, X, y, n_to_select):
        """Initializes the search. Prepares an array to store the selections,
        makes the initial selection (unless provided), and computes the starting
        hausdorff distances.
        """
        super()._init_greedy_search(X, y, n_to_select)

        self.norms_ = (X**2).sum(axis=abs(self._axis - 1))
        self.hausdorff_ = np.full(X.shape[self._axis], np.inf)
        self.hausdorff_at_select_ = np.full(X.shape[self._axis], np.inf)

        if isinstance(self.initialize, (np.ndarray, list)):
            if all(isinstance(i, numbers.Integral) for i in self.initialize):
                for i, val in enumerate(self.initialize):
                    self.selected_idx_[i] = val
                    self._update_post_selection(X, y, self.selected_idx_[i])
            else:
                raise ValueError("Invalid value of the initialize parameter")
        elif self.initialize == "random":
            random_state = check_random_state(self.random_state)
            initialize = random_state.randint(X.shape[self._axis])
            self.selected_idx_[0] = initialize
            self._update_post_selection(X, y, self.selected_idx_[0])
        elif isinstance(self.initialize, numbers.Integral):
            initialize = self.initialize
            self.selected_idx_[0] = initialize
            self._update_post_selection(X, y, self.selected_idx_[0])
        else:
            raise ValueError("Invalid value of the initialize parameter")

    def _update_hausdorff(self, X, y, last_selected):
        self.hausdorff_at_select_[last_selected] = self.hausdorff_[last_selected]

        # distances of all points to the new point
        if self._axis == 1:
            new_dist = (
                self.norms_ + self.norms_[last_selected] - 2 * X[:, last_selected].T @ X
            )
        else:
            new_dist = (
                self.norms_ + self.norms_[last_selected] - 2 * X[last_selected] @ X.T
            )

        # update in-place the Hausdorff distance list
        np.minimum(self.hausdorff_, new_dist, self.hausdorff_)

    def _update_post_selection(self, X, y, last_selected):
        """
        Saves the most recent selections, increments the counter,
        and, recomputes hausdorff distances.
        """
        self._update_hausdorff(X, y, last_selected)
        super()._update_post_selection(X, y, last_selected)


class _PCovFPS(GreedySelector):
    r"""Transformer that performs Greedy Selection using PCovR-weighted
    Farthest Point Sampling. In PCov-FPS, a modified covariance or Gram matrix is used
    to express the distances.

    For sample selection, this is a modified kernel matrix.

    Parameters
    ----------
    mixing: float, default=0.5
            The PCovR mixing parameter, as described in PCovR as
            :math:`{\alpha}`
    initialize: int or 'random', default=0
        Index of the first selection. If 'random', picks a random
        value when fit starts.
    """

    def __init__(
        self,
        selection_type,
        mixing=0.5,
        initialize=0,
        n_to_select=None,
        score_threshold=None,
        score_threshold_type="absolute",
        progress_bar=False,
        full=False,
        random_state=0,
    ):
        if mixing == 1.0:
            raise ValueError(
                "Mixing = 1.0 corresponds to traditional FPS."
                "Please use the FPS class."
            )

        self.mixing = mixing
        self.initialize = initialize

        super().__init__(
            selection_type=selection_type,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            score_threshold_type=score_threshold_type,
            progress_bar=progress_bar,
            full=full,
            random_state=random_state,
        )

    def score(self, X, y=None):
        """Returns the Hausdorff distances of all samples to previous selections.

        NOTE: This function does not compute the importance score each time it
        is called, in order to avoid unnecessary computations. The hausdorff
        distance is updated in :py:func:`self._update_hausdorff`

        Parameters
        ----------
        X : ignored
        y : ignored

        Returns
        -------
        hausdorff : Hausdorff distances
        """
        return self.hausdorff_

    def get_distance(self):
        """
        Returns
        -------
        hausdorff : numpy.ndarray of shape (`n_to_select_from_`)
            the minimum distance from each point to the set of selected points. once a
            point is selected, the distance is not updated; the final list will reflect
            the distances when selected.
        """
        return self.hausdorff_

    def get_select_distance(self):
        """
        Returns
        -------
        hausdorff_at_select : numpy.ndarray of shape (`n_to_select`)
            at the time of selection, the minimum distance from each selected point to
            the set of previously selected points.
        """
        mask = self.get_support(indices=True, ordered=True)
        return self.hausdorff_at_select_[mask]

    def _init_greedy_search(self, X, y, n_to_select):
        """Initializes the search. Prepares an array to store the selections,
        makes the initial selection (unless provided), and computes the starting
        hausdorff distances.
        """
        super()._init_greedy_search(X, y, n_to_select)

        if self._axis == 1:
            self.pcovr_distance_ = pcovr_covariance(mixing=self.mixing, X=X, Y=y)
        else:
            self.pcovr_distance_ = pcovr_kernel(mixing=self.mixing, X=X, Y=y)

        self.norms_ = np.diag(self.pcovr_distance_)

        if self.initialize == "random":
            random_state = check_random_state(self.random_state)
            initialize = random_state.randint(X.shape[self._axis])
        elif isinstance(self.initialize, numbers.Integral):
            initialize = self.initialize
        else:
            raise ValueError("Invalid value of the initialize parameter")

        self.selected_idx_[0] = initialize
        self.hausdorff_ = np.full(X.shape[self._axis], np.inf)
        self.hausdorff_at_select_ = np.full(X.shape[self._axis], np.inf)
        self._update_post_selection(X, y, self.selected_idx_[0])

    def _update_hausdorff(self, X, y, last_selected):
        self.hausdorff_at_select_[last_selected] = self.hausdorff_[last_selected]

        # distances of all points to the new point
        new_dist = (
            self.norms_
            + self.norms_[last_selected]
            - 2 * np.take(self.pcovr_distance_, last_selected, axis=self._axis)
        )

        # update in-place the Hausdorff distance list
        np.minimum(self.hausdorff_, new_dist, self.hausdorff_)

    def _update_post_selection(self, X, y, last_selected):
        """Saves the most recent selections, increments the counter, and, recomputes
        hausdorff distances.
        """
        self._update_hausdorff(X, y, last_selected)
        super()._update_post_selection(X, y, last_selected)

    def _more_tags(self):
        """Pass that this method requires a target vector"""
        return {
            "requires_y": True,
        }
