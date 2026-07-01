"""Contains custom types and implemented options for the detectors."""

from typing import Literal, TypeAlias, TypeVar

import numpy as np
import torch
from numpy.typing import DTypeLike as NumpyDTypeLike

# The type of the main manipulated object.
ArrayLike = TypeVar("ArrayLike", np.ndarray, torch.Tensor)  #: The type of the main manipulated object.
DTypeLike = TypeVar("DTypeLike", NumpyDTypeLike, torch.dtype)  #: The data type of the main manipulated object.

# Types aliases
ShapeType: TypeAlias = int | tuple[int, ...]  #: Alias for the type of the shape of an array.
Number: TypeAlias = float | int  #: Alias for the type of a number.

IMPLEMENTED_BACKEND = Literal["numpy", "torch"]
"""The implemented backends.

:const:`numpy`

A backend using Numpy and Scipy.

:const:`torch`

A backend using torch with `cpu` and `GPU` implementations.
"""

# Implemented options for commons classes
IMPLEMENTED_DISTANCES = Literal["euclidean", "mahalanobis"]
"""The implemented distance metrics.

:const:`euclidean`

.. math::

    D_{i,j} = d_E(X_i, Y_j) = \\sqrt{(X_i - Y_j)^T (X_i - Y_j)}^p.

:const:`mahalanobis`

.. math::

    D_{i,j} = d_M(X_i, Y_j) = \\sqrt{(X_i - Y_j)^T \\Sigma^{-1} (X_i - Y_j)}^p.
"""
IMPLEMENTED_INCREMENTERS = Literal["inverse", "sherman", "woodbury"]
"""The implemented incrementer methods.

:const:`inverse`

    Here M is the moment matrix.

    .. math::

        & N_X = |X| \\\\
        & V = \\begin{pmatrix}
            P_0(X_0) & \\cdots & P_n(X_0) \\\\
            \\vdots & \\ddots & \\vdots \\\\
            P_0(X_{N_X}) & \\cdots & P_n(X_{N_X})
            \\end{pmatrix} \\\\
        & M_{updated} = ((N \\times M) + V^T ~ V) / (N + N_X) \\\\
        & M^{-1}_{updated} = \\text{inv}(M_{updated})

:const:`sherman`
        
    Here M is the inverse of the moment matrix.

    .. math::
        
        & N_X = |X| \\\\
        & M^{(0)} = M / N \\\\
        & \\text{For } i = 1 \\cdots N_X: \\\\
            & \\qquad V^{(i)} = \\left(P_0(X_i), \\cdots, P_n(X_i) \\right)^T \\\\
            & \\qquad M^{(i)} = M^{(i-1)} - \\frac{M^{(i-1)} V^{(i)} V^{(i)^T} M^{(i-1)}}{1 + V^{(i)^T} M^{(i-1)} V^{(i)}} \\\\
        & M^{-1}_{updated} = M^{(N_X)} \\times (N + N_X)

:const:`woodbury`

    Here M is the inverse of the moment matrix.

    .. math::

        & N_X = |X| \\\\
        & V = \\begin{pmatrix}
            P_0(X_0) & \\cdots & P_n(X_0) \\\\
            \\vdots & \\ddots & \\vdots \\\\
            P_0(X_{N_X}) & \\cdots & P_n(X_{N_X})
            \\end{pmatrix} \\\\
        & M^{-1}_{updated} = M - M V \\left( I + V^T M V \\right)^{-1} V^T M
"""
IMPLEMENTED_INVERTERS = Literal["inv", "pseudo", "solve", "fpd"]
"""The implemented inverters methods.

:const:`inv`

Compute the classical inversion implemented in :attr:`backend`.

:const:`pseudo`

Compute the (Moore-Penrose) pseudo-inverse.

:const:`solve`

Compute the inverse by solving the system :math:`X X^{-1} = I` to obtain :math:`X^{-1}`.

:const:`fpd`

Compute the fast positive definite inverse of matrix using Cholesky decomposition: :math:`X^{-1} = L^{-T} L^{-1}` where :math:`X = L L^T`.
"""
IMPLEMENTED_POLYNOMIAL_BASIS = Literal["monomials", "chebyshev"]
"""The implemented polynomial basis.

:const:`monomials`

.. math::

    P_k(x) = x^k.

:const:`chebyshev`

.. math::

    P_k(x) = T_k(x) = \\begin{cases} & cos(n ~ arccos(x)) & \\text{if } |x| < 1 \\\\ & cosh(n ~ arccosh(x)) & \\text{if } x \\geq 1 \\\\ & (-1)^n cosh(n ~ arccosh(-x)) & \\text{if } x \\leq -1 \\end{cases}.
"""
IMPLEMENTED_SOLVERS = Literal["cholesky", "qr", "inverse", "solve"]
"""The implemented solvers.

The objective is to solve :math:`z = v^T G^{-1} v` with :math:`G = (V^T V) / N`.

For all methods, :math:`x = A^{-1} b` is computed by solving :math:`A x = b`. 

:const:`cholesky`

.. math::

    & \\text{Find } L \\text{ such that } G = L L^T \\text{ with } L \\text{ a lower triangular matrix} \\\\
    & y = L^{-1} v \\\\
    & x = L^{-T} y \\\\
    & z = v^T x \\\\

:const:`qr`

.. math::

    & \\text{Find } R \\text{ such that } V = Q R \\text{ with } Q \\text{ an orthogonal matrix and } R \\text{ an upper triangular matrix} \\\\
    & y = (R^{-T} v) / N \\\\
    & x = R^{-1} y \\\\
    & z = v^T x

:const:`inverse`

.. math::

    & G_{inv} = G^{-1} I \\\\
    & x = G_{inv} v \\\\
    & z = v^T x

:const:`solve`

.. math::

    & x = G^{-1} v \\\\
    & z = v^T x

"""
IMPLEMENTED_STORAGES = Literal["full", "batch"]
"""The implemented storage methods.

:const:`full`

:attr:`batch_size` is set to :math:`|X|`, thus returning the whole matrix in one iteration.

:const:`batch`

Loop over the first dimension of :math:`X` returning matrices of shape (:attr:`batch_size`, ...).

.. note ::

    The dimension of the last iteration can be lower than :attr:`batch_size`.
"""
IMPLEMENTED_THRESHOLD_SCHEMES = Literal["constant", "comb", "vu", "vuC"]
"""The implemented threshold schemes.

:const:`constant`

.. math::

    C

:const:`comb`

.. math::

    \\begin{pmatrix} n+d \\\\ n \\end{pmatrix}

:const:`vu`

.. math::

    n^{3d / 2}

:const:`vuC`

.. math::

    \\frac{1}{C} n^{3d / 2}
"""
