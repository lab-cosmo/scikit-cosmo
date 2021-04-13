import numpy as np
from scipy import linalg
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.extmath import randomized_svd

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
from copy import deepcopy


def check_lr_fit(regressor, X, y=None):
    r"""
    Checks that an regressor is fitted, and if not,
    fits it with the provided data

    :param regressor: sklearn-style regressor
    :type regressor: object
    :param X: feature matrix with which to fit the regressor
        if it is not already fitted
    :type X: array
    :param y: target values with which to fit the regressor
        if it is not already fitted
    :type y: array
    :param sample_weight: sample weights with which to fit
        the regressor if not already fitted
    :type sample_weight: array of shape (n_samples,)
    """
    try:
        check_is_fitted(regressor)
        fitted_regressor = deepcopy(regressor)
    except NotFittedError:
        fitted_regressor = clone(regressor)
        fitted_regressor.fit(X, y=y)

    return fitted_regressor


def pcovr_covariance(
    mixing,
    X,
    Y,
    rcond=1e-12,
    return_isqrt=False,
    rank=None,
    random_state=0,
    iterated_power=None,
):
    r"""
    Creates the PCovR modified covariance

    .. math::

        \mathbf{\tilde{C}} = \alpha \mathbf{X}^T \mathbf{X} +
        (1 - \alpha) \left(\left(\mathbf{X}^T
        \mathbf{X}\right)^{-\frac{1}{2}} \mathbf{X}^T
        \mathbf{\hat{Y}}\mathbf{\hat{Y}}^T \mathbf{X} \left(\mathbf{X}^T
        \mathbf{X}\right)^{-\frac{1}{2}}\right)

    where :math:`\mathbf{\hat{Y}}`` are the properties obtained by linear regression.

    :param mixing: mixing parameter,
                   as described in PCovR as :math:`{\alpha}`, defaults to 1
    :type mixing: float

    :param X: Data matrix :math:`\mathbf{X}`
    :type X: array of shape (n x m)

    :param Y: array to include in biased selection when mixing < 1
    :type Y: array of shape (n x p)

    :param rcond: threshold below which eigenvalues will be considered 0,
                      defaults to 1E-12
    :type rcond: float

    :param return_isqrt: Whether to return the calculated inverse square root of
                         the covariance. Used when inverse square root is needed
                         and the pcovr_covariance has already been calculated
    :type return_isqrt: boolean

    :param rank: number of eigenpairs to estimate the inverse square root
                 with. Defaults to min(X.shape)
    :type rank: int

    :param random_state: random seed to use for randomized svd
    :type random_state: int

    """

    C = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)

    if mixing < 1 or return_isqrt:

        if rank is None:
            rank = min(X.shape)

        if rank >= min(X.shape):
            _, vC, UC = linalg.svd(X, full_matrices=False)
        else:
            _, vC, UC = randomized_svd(
                X,
                n_components=rank,
                n_iter=iterated_power,
                flip_sign=True,
                random_state=random_state,
            )

        UC = UC.T[:, vC > rcond]
        vC = vC[vC > rcond]

        C_isqrt = UC @ np.diagflat(1.0 / vC) @ UC.T

        # parentheses speed up calculation greatly
        C_Y = C_isqrt @ (X.T @ Y)
        C_Y = C_Y.reshape((C.shape[0], -1))
        C_Y = np.real(C_Y)

        C += (1 - mixing) * C_Y @ C_Y.T

    if mixing > 0:
        C += (mixing) * (X.T @ X)

    if return_isqrt:
        return C, C_isqrt
    else:
        return C


def pcovr_kernel(mixing, X, Y, **kernel_params):
    r"""
    Creates the PCovR modified kernel distances

    .. math::

        \mathbf{\tilde{K}} = \alpha \mathbf{K} +
        (1 - \alpha) \mathbf{Y}\mathbf{Y}^T

    the default kernel is the linear kernel, such that:

    .. math::

        \mathbf{\tilde{K}} = \alpha \mathbf{X} \mathbf{X}^T +
        (1 - \alpha) \mathbf{Y}\mathbf{Y}^T

    :param mixing: mixing parameter,
                   as described in PCovR as :math:`{\alpha}`, defaults to 1
    :type mixing: float

    :param X: Data matrix :math:`\mathbf{X}`
    :type X: array of shape (n x m)

    :param Y: array to include in biased selection when mixing < 1
    :type Y: array of shape (n x p)

    :param kernel_params: dictionary of arguments to pass to pairwise_kernels
                         if none are specified, assumes that the kernel is linear
    :type kernel_params: dictionary, optional

    """

    K = np.zeros((X.shape[0], X.shape[0]))
    if mixing < 1:
        K += (1 - mixing) * Y @ Y.T
    if mixing > 0:
        if "kernel" not in kernel_params:
            K += (mixing) * X @ X.T
        elif kernel_params.get("kernel") != "precomputed":
            K += (mixing) * pairwise_kernels(X, **kernel_params)
        else:
            K += (mixing) * X

    return K
