import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing._data import KernelCenterer
from sklearn.utils.validation import FLOAT_DTYPES, _check_sample_weight, check_is_fitted


class StandardFlexibleScaler(TransformerMixin, BaseEstimator):
    """Standardize features by removing the mean and scaling to unit variance.
    Reduce the mean of the column to zero and, in the case of `column_wise=True`
    the variance of each column equal to one / number of columns.
    The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the samples if `with_mean`, otherwise zero,
    and `s` is the standard deviation of the samples if `with_std` or one.

    Centering and scaling can occur independently for each feature by calculating
    the appropriate statistics for the input or for the
    whole matrix (`column_wise=False`). The mean and standard deviation are then
    stored for use on later data using :py:meth:`transform`.

    Standardization of a dataset is a common requirement for many
    machine learning estimators: an improperly scaled / centered
    dataset may result in anomalous behavior.

    At the same time, depending on the conditions of the task, it may be necessary
    to preserve the ratio in the scale between the features (for example, in the
    case where the feature matrix is something like a covariance matrix), so
    the standardization should be carried out for the whole matrix,
    as opposed to the individual columns,
    as is done in `sklearn.preprocessing.StandardScaler`.

    Parameters
    ----------
    with_mean: bool, default=True
        If True, center the data before scaling. If False, keep the mean intact

    with_std: bool, default=True
        If True, scale the data to unit variance. If False, keep the variance intact

    column_wise: bool, default=False
        If True, normalize each column separately. If False, normalize the whole
        matrix with respect to its total variance.

    rtol: float, default=0
        The relative tolerance for the optimization: variance is
        considered zero when it is less than abs(mean) * rtol + atol.

    atol: float, default=1.0E-12
        The relative tolerance for the optimization: variance is
        considered zero when it is less than abs(mean) * rtol + atol.

    copy : bool, default=None
        Copy the input X or not.

    Attributes
    ----------
    n_samples_in_: int
        Number of samples in the reference ndarray
    n_features_in_: int
        Number of features in the reference ndarray
    mean_ : numpy.ndarray of shape (n_features,)
        The mean value for each feature in the training set. Equal to
        :class:`numpy.ndarray` of zeros shape (n_features,) when ``with_mean=False``.
    scale_ : numpy.ndarray of shape (n_features,), float  or None
        The scaling factor, :class:`numpy.ndarray` of shape (n_features,)
        when `column_wise=True` or float  when `column_wise = False`.
    copy : bool, default=None
        Copy the input X or not.

    Examples
    --------
    >>> import numpy as np
    >>> from skmatter.preprocessing import StandardFlexibleScaler
    >>> X = np.array([[1.0, -2.0, 2.0], [-2.0, 1.0, 3.0], [4.0, 1.0, -2.0]])
    >>> transformer = StandardFlexibleScaler().fit(X)
    >>> transformer
    StandardFlexibleScaler()
    >>> transformer.transform(X)
    array([[ 0.        , -0.56195149,  0.28097574],
           [-0.84292723,  0.28097574,  0.56195149],
           [ 0.84292723,  0.28097574, -0.84292723]])
    >>> transformer.scale_ * transformer.transform(X)
    array([[ 0., -2.,  1.],
           [-3.,  1.,  2.],
           [ 3.,  1., -3.]])
    >>> transformer.scale_ * transformer.transform(X) + transformer.mean_
    array([[ 1., -2.,  2.],
           [-2.,  1.,  3.],
           [ 4.,  1., -2.]])
    """

    def __init__(
        self,
        with_mean=True,
        with_std=True,
        column_wise=False,
        rtol=0,
        atol=1e-12,
        copy=False,
    ):
        """Initialize StandardFlexibleScaler."""
        self.with_mean = with_mean
        self.with_std = with_std
        self.column_wise = column_wise
        self.rtol = rtol
        self.atol = atol
        self.copy = copy

    def fit(self, X, y=None, sample_weight=None):
        """Compute mean and scaling to be applied for subsequent normalization.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y: None
            Ignored.
        sample_weight: numpy.ndarray of shape (n_samples,)
            Weights for each sample. Sample weighting can be used to center
            (and scale) data using a weighted mean. Weights are internally
            normalized before preprocessing.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        X = self._validate_data(
            X,
            copy=self.copy,
            estimator=self,
            dtype=FLOAT_DTYPES,
            ensure_min_samples=2,
        )
        self.n_samples_in_, self.n_features_in_ = X.shape

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
            sample_weight = sample_weight / np.sum(sample_weight)

        if self.with_mean:
            self.mean_ = np.average(X, weights=sample_weight, axis=0)
        else:
            self.mean_ = np.zeros(self.n_features_in_)

        self.scale_ = 1.0
        if self.with_std:
            X_mean = np.average(X, weights=sample_weight, axis=0)
            var = np.average((X - X_mean) ** 2, weights=sample_weight, axis=0)

            if self.column_wise:
                if np.any(var < self.atol + abs(X_mean) * self.rtol):
                    raise ValueError("Cannot normalize a feature with zero variance")
                self.scale_ = np.sqrt(var)
            else:
                var_sum = var.sum()
                if var_sum < abs(np.average(X_mean)) * self.rtol + self.atol:
                    raise ValueError("Cannot normalize a matrix with zero variance")
                self.scale_ = np.sqrt(var_sum)

        return self

    def transform(self, X, y=None, copy=None):
        """Normalize a vector based on previously computed mean and scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis.
        y: None
            Ignored.
        copy : bool, default=None
            Copy the input X or not.

        Returns
        -------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        copy = copy if copy is not None else self.copy
        X = self._validate_data(
            X,
            reset=False,
            copy=copy,
            estimator=self,
            dtype=FLOAT_DTYPES,
        )
        check_is_fitted(
            self, attributes=["n_samples_in_", "n_features_in_", "scale_", "mean_"]
        )

        if self.n_features_in_ != X.shape[1]:
            raise ValueError("X shape does not match training shape")
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X_tr):
        """Scale back the data to the original representation.

        Parameters
        ----------
        X_tr : numpy.ndarray of shape (n_samples, n_features)
            Transformed matrix

        Returns
        -------
        X : original matrix
        """
        check_is_fitted(
            self, attributes=["n_samples_in_", "n_features_in_", "scale_", "mean_"]
        )

        if self.n_features_in_ != X_tr.shape[1]:
            raise ValueError("X shape does not match training shape")
        return X_tr * self.scale_ + self.mean_


class KernelNormalizer(KernelCenterer):
    r"""Kernel centering method, similar to KernelCenterer,
    but with additional scaling and ability to pass a set of sample weights.

    Let :math:`K(x, z)` be a kernel defined by :math:`\phi(x)^T \phi(z)`,
    where :math:`\phi` is a function mapping x to a Hilbert space.
    KernelNormalizer centers (i.e., normalize to have zero mean) the data without
    explicitly computing :math:`\phi(x)`.
    It is equivalent to centering and scaling :math:`\phi(x)` with
    sklearn.preprocessing.StandardScaler(with_std=False).

    Parameters
    ----------
    with_center: bool, default=True
        If True, center the kernel matrix before scaling. If False, do not
        center the kernel
    with_trace: bool, default=True
        If True, scale the kernel so that the trace is equal to the number of
        samples. If False, do not scale the kernel

    Attributes
    ----------
    K_fit_rows_ : numpy.ndarray of shape (n_samples,)
        Average of each column of kernel matrix.
    K_fit_all_ : float
        Average of kernel matrix.
    sample_weight_ : float
        Sample weights (if provided during the fit)
    scale_ : float
        Scaling parameter used when 'with_trace'=True
        Calculated as np.trace(K) / K.shape[0]

    Examples
    --------
    >>> from skmatter.preprocessing import KernelNormalizer
    >>> from sklearn.metrics.pairwise import pairwise_kernels
    >>> X = [[1.0, -2.0, 2.0], [-2.0, 1.0, 3.0], [4.0, 1.0, -2.0]]
    >>> K = pairwise_kernels(X, metric="linear")
    >>> K
    array([[  9.,   2.,  -2.],
           [  2.,  14., -13.],
           [ -2., -13.,  21.]])
    >>> transformer = KernelNormalizer().fit(K)
    >>> transformer
    KernelNormalizer()
    >>> transformer.transform(K)
    array([[ 0.39473684,  0.        , -0.39473684],
           [ 0.        ,  1.10526316, -1.10526316],
           [-0.39473684, -1.10526316,  1.5       ]])
    >>> transformer.scale_ * transformer.transform(K)
    array([[  5.,   0.,  -5.],
           [  0.,  14., -14.],
           [ -5., -14.,  19.]])
    >>>
    """

    def __init__(self, with_center=True, with_trace=True):
        """Initialize KernelNormalizer."""
        self.with_center = with_center
        self.with_trace = with_trace
        super().__init__()

    def fit(self, K, y=None, sample_weight=None):
        """Fit KernelFlexibleCenterer

        Parameters
        ----------
        K : numpy.ndarray of shape (n_samples, n_samples)
            Kernel matrix.
        y : None
            Ignored.
        sample_weight: numpy.ndarray of shape (n_samples,), default=None
            Weights for each sample. Sample weighting can be used to center (and scale)
            data using a weighted mean. Weights are internally normalized before
            preprocessing.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        K = self._validate_data(K, copy=True, dtype=FLOAT_DTYPES, reset=False)

        if sample_weight is not None:
            self.sample_weight_ = _check_sample_weight(sample_weight, K, dtype=K.dtype)
            self.sample_weight_ = self.sample_weight_ / np.sum(self.sample_weight_)
        else:
            self.sample_weight_ = sample_weight

        if self.with_center:
            if self.sample_weight_ is not None:
                self.K_fit_rows_ = np.average(K, weights=self.sample_weight_, axis=0)
                self.K_fit_all_ = np.average(
                    self.K_fit_rows_, weights=self.sample_weight_
                )
            else:
                super().fit(K, y)

            K_pred_cols = np.average(K, weights=self.sample_weight_, axis=1)[
                :, np.newaxis
            ]
        else:
            self.K_fit_rows_ = np.zeros(K.shape[1])
            self.K_fit_all_ = 0.0
            K_pred_cols = np.zeros((K.shape[0], 1))

        if self.with_trace:
            K -= self.K_fit_rows_
            K -= K_pred_cols
            K += self.K_fit_all_

            self.scale_ = np.trace(K) / K.shape[0]
        else:
            self.scale_ = 1.0

        return self

    def transform(self, K, copy=True):
        """Center kernel matrix.

        Parameters
        ----------
        K : numpy.ndarray of shape (n_samples1, n_samples2)
            Kernel matrix.
        copy : bool, default=True
            Set to False to perform inplace computation.

        Returns
        -------
        K_new : numpy.ndarray of shape (n_samples1, n_samples2)
            Transformed array
        """
        check_is_fitted(self)
        K = self._validate_data(K, copy=copy, dtype=FLOAT_DTYPES, reset=False)

        if self.with_center:
            K_pred_cols = np.average(K, weights=self.sample_weight_, axis=1)[
                :, np.newaxis
            ]
        else:
            K_pred_cols = np.zeros((K.shape[0], 1))

        K -= self.K_fit_rows_
        K -= K_pred_cols
        K += self.K_fit_all_

        return K / self.scale_

    def fit_transform(self, K, y=None, sample_weight=None, copy=True, **fit_params):
        r"""Fit to data, then transform it.

        Parameters
        ----------
        K : numpy.ndarray of shape (n_samples, n_samples)
            Kernel matrix.
        y : None
            Ignored.
        sample_weight: numpy.ndarray of shape (n_samples,), default=None
            Weights for each sample. Sample weighting can be used to center (and
            scale) data using a weighted mean. Weights are internally normalized
            before preprocessing.
        \**fit_params:
            necessary for compatibility with the functions of the
            TransformerMixin class

        Returns
        -------
        K_new : numpy.ndarray of shape (n_samples1, n_samples2)
            Transformed array
        """
        self.fit(K, y, sample_weight=sample_weight)
        return self.transform(K, copy)


class SparseKernelCenterer(TransformerMixin):
    r"""Kernel centering method for sparse kernels, similar to
    :class:`KernelFlexibleCenterer`.

    The main disadvantage of kernel methods, which is widely used in machine
    learning it is that they quickly grow in time and space complexity with the
    number of sample. It is clear that with a large dataset, not only do you
    need to store a huge amount of information, but you also need to use it
    constantly in calculations. In order to avoid this, so-called sparse kernel
    methods are used formulated from the low-dimensional (The Nystrom)
    approximation:

    .. math::
        \mathbf{K} \approx \hat{\mathbf{K}}_{N N}
            = \mathbf{K}_{N M} \mathbf{K}_{M M}^{-1} \mathbf{K}_{N M}^{T}

    where the subscripts for $\mathbf{K}$ denote the size of the sets of samples
    compared in each kernel, with $N$ being the size of the full data set and
    $M$ referring a small, active set containing $M$ samples. With this
    method it is only need to save and use the matrix $\mathbf{K}_{NM}$, i.e. it
    is possible to get a $N/M$ times improvement in the asymptotic by memory.

    Parameters
    ----------
    with_center: bool, default=True
        If True, center the kernel matrix before scaling. If False, do not
        center the kernel
    with_trace : bool, default=True
        If True, scale the kernel so that the trace is equal to the number of
        samples. If False, do not scale the kernel
    rcond : float, default 1E-12
        conditioning parameter to use when computing the Nystrom-approximated
        kernel for scaling

    Attributes
    ----------
    K_fit_rows_ : numpy.ndarray of shape (n_samples,)
        Average of each column of kernel matrix.
    K_fit_all_ : float
        Average of kernel matrix.
    sample_weight_ : float
        Sample weights (if provided during the fit)
    scale_ : float
        Scaling parameter used when 'with_trace'=True
        Calculated as np.trace(K) / K.shape[0]
    n_active_: int
        size of active set
    """

    def __init__(self, with_center=True, with_trace=True, rcond=1e-12):
        self.with_center = with_center
        self.with_trace = with_trace
        self.rcond = rcond

    def fit(self, Knm, Kmm, y=None, sample_weight=None):
        """Fit ``KernelFlexibleCenterer``

        Parameters
        ----------
        Knm : numpy.ndarray of shape (n_samples, n_active)
            Kernel matrix between the reference data set and the active set
        Kmm : numpy.ndarray of shape (n_active, n_active)
            Kernel matrix between the active set and itself
        y : None
            Ignored.
        sample_weight: numpy.ndarray of shape (n_samples,), default=None
            Weights for each sample. Sample weighting can be used to center (and
            scale) data using a weighted mean. Weights are internally normalized
            before preprocessing.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        if Knm.shape[1] != Kmm.shape[0]:
            raise ValueError(
                "The reference kernel is not commensurate shape with the "
                "active kernel."
            )

        if Kmm.shape[0] != Kmm.shape[1]:
            raise ValueError("The active kernel is not square.")

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, Knm, dtype=Knm.dtype)
            sample_weight = sample_weight / np.sum(sample_weight)

        self.n_active_ = Kmm.shape[0]

        if self.with_center:
            self.K_fit_rows_ = np.average(Knm, weights=sample_weight, axis=0)
        else:
            self.K_fit_rows_ = np.zeros(Knm.shape[1])

        if self.with_trace:
            Knm_centered = Knm - self.K_fit_rows_

            # The following is more correctly written as Knm @ Kmm^{-1} @ Knm.T
            # but has been changed to Knm.T @ Knm @ Kmm^{-1} to avoid the memory
            # overload often caused by storing n x n matrices. This is fine
            # for the following trace, but should not be used for other operations.
            Khat_trace = np.trace(
                Knm_centered.T @ Knm_centered @ np.linalg.pinv(Kmm, self.rcond)
            )

            self.scale_ = Khat_trace / Knm.shape[0]
        else:
            self.scale_ = 1.0

        return self

    def transform(self, Knm, y=None):
        """Centering our Kernel. Previously you should fit data.

        Parameters
        ----------
        Knm: numpy.ndarray of shape (n_samples, n_active)
            Kernel matrix between the reference data set and the active set
        y : None
            Ignored.

        Returns
        -------
        K_new : numpy.ndarray of shape (n_samples, n_active)
            Transformed array
        """
        check_is_fitted(self, attributes=["scale_", "K_fit_rows_", "n_active_"])

        if Knm.shape[1] != self.n_active_:
            raise ValueError(
                "The reference kernel and received kernel have different shape"
            )

        Kc = (Knm - self.K_fit_rows_) / self.scale_

        return Kc

    def fit_transform(self, Knm, Kmm, y=None, sample_weight=None, **fit_params):
        r"""Fit to data, then transform it.

        Parameters
        ----------
        Knm: numpy.ndarray of shape (n_samples, n_active)
            Kernel matrix between the reference data set and the active set
        Kmm: numpy.ndarray of shape (n_active, n_active)
            Kernel matrix between the active set and itself
        y : None
            Ignored.
        sample_weight: numpy.ndarray of shape (n_samples,), default=None
            Weights for each sample. Sample weighting can be used to center (and
            scale) data using a weighted mean. Weights are internally normalized
            before preprocessing.
        \**fit_params:
            necessary for compatibility with the functions of the
            TransformerMixin class

        Returns
        -------
        K_new : numpy.ndarray of shape (n_samples, n_active)
            Transformed array
        """
        self.fit(Knm=Knm, Kmm=Kmm, sample_weight=sample_weight)
        return self.transform(Knm)
