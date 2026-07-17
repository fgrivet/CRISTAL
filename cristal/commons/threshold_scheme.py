"""Contains the :class:`ThresholdScheme <cristal.commons.threshold_scheme.ThresholdScheme>` class used in detectors."""

from math import comb
from typing import get_args

from ..types import IMPLEMENTED_THRESHOLD_SCHEMES, Number
from .base_commons import BaseCommons


# pylint: disable=unused-variable
class ThresholdScheme(BaseCommons):
    """Class to compute a threshold based on data properties.

    Parameters
    ----------
    scheme : IMPLEMENTED_THRESHOLD_SCHEMES, optional
        The threshold scheme to use, by default "constant"
    C : Number
        A constant (potentially) used during the computation of the threshold.

    Attributes
    ----------
    scheme : :class:`IMPLEMENTED_THRESHOLD_SCHEMES <cristal.types.IMPLEMENTED_THRESHOLD_SCHEMES>`
        The threshold scheme to use.
    C : Number
        A constant (potentially) used during the computation of the threshold.

    Raises
    ------
    ValueError
        If the threshold :const:`scheme` is not valid.

    See Also
    --------
    cristal.types.IMPLEMENTED_THRESHOLD_SCHEMES : For more details on how the threshold schemes work.

    Examples
    --------
    >>> threshold_scheme = ThresholdScheme(scheme="constant", C=1)
    >>> threshold = threshold_scheme(n, d)
    """

    def __init__(self, scheme: IMPLEMENTED_THRESHOLD_SCHEMES = "constant", C: Number = 1):
        """Class constructor.
        Define the threshold :attr:`scheme`.

        Parameters
        ----------
        scheme : IMPLEMENTED_THRESHOLD_SCHEMES, optional
            The threshold scheme to use, by default "constant"
        C : Number
            A constant (potentially) used during the computation of the threshold.

        Raises
        ------
        ValueError
            If the threshold :const:`scheme` is not valid.

        See Also
        --------
        cristal.types.IMPLEMENTED_THRESHOLD_SCHEMES : For more details on how the threshold schemes work.
        """
        if scheme not in get_args(IMPLEMENTED_THRESHOLD_SCHEMES):
            raise ValueError(f"scheme must be in {IMPLEMENTED_THRESHOLD_SCHEMES}. Got {scheme}.")
        self.scheme: IMPLEMENTED_THRESHOLD_SCHEMES = scheme
        """The threshold scheme to use.

        See Also
        --------
        cristal.types.IMPLEMENTED_THRESHOLD_SCHEMES : For more details on how the threshold schemes work.
        """

        self.C: Number = C
        """A constant (potentially) used during the computation of the threshold."""

    def compute_threshold(self, n: int | float, d: int) -> float:
        """Compute a threshold based on data properties.

        Parameters
        ----------
        n : int | float
            The maximum degree of polynomials.
        d : int
            The dimension of data.

        Returns
        -------
        float
            The threshold associated to the data and the corresponding :attr:`scheme`.

        Examples
        --------
        >>> threshold_scheme = ThresholdScheme(scheme="constant", C=1)
        >>> threshold = threshold_scheme(n, d)
        """
        # Comb
        if self.scheme == "comb":
            if isinstance(n, float):
                n = int(n)
            return comb(d + n, n)
        # Vu / VuC
        if self.scheme in ["vu", "vuC"]:
            threshold = n ** (3 * d / 2)
            if self.scheme == "vuC":
                threshold /= self.C
            return threshold
        # Constant
        return self.C

    def __call__(self, n: int | float, d: int) -> float:
        """Compute a threshold based on data properties.

        .. hint::

            This function is a wrapper for :func:`compute_threshold`.

        Parameters
        ----------
        n : int | float
            The maximum degree of polynomials.
        d : int
            The dimension of data.

        Returns
        -------
        float
            The threshold associated to the data and the corresponding :attr:`scheme`.
        """
        return self.compute_threshold(n=n, d=d)
