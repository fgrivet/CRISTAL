from math import comb
from typing import Literal, get_args

from .base_commons import BaseCommons

IMPLEMENTED_THRESHOLD_SCHEME = Literal["constant", "comb", "vu", "vuC"]


class ThresholdScheme(BaseCommons):

    def __init__(self, scheme: IMPLEMENTED_THRESHOLD_SCHEME = "constant"):
        if scheme not in get_args(IMPLEMENTED_THRESHOLD_SCHEME):
            raise ValueError(f"scheme must be in {IMPLEMENTED_THRESHOLD_SCHEME}. Got {scheme}.")
        self.scheme = scheme

    def compute_threshold(self, n: int | float, d: int, C: float | int) -> float:
        # Comb
        if self.scheme == "comb":
            if isinstance(n, float):
                n = int(n)
            return comb(d + n, n)
        # Vu / VuC
        if self.scheme in ["vu", "vuC"]:
            threshold = n ** (3 * d / 2)
            if self.scheme == "vuC":
                threshold /= C
            return threshold
        # Constant
        return C

    def __call__(self, n: int | float, d: int, C: float | int) -> float:
        return self.compute_threshold(n=n, d=d, C=C)
