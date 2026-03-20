from typing import TypeAlias, TypeVar, Union, Literal

import numpy as np
import torch
from numpy.typing import DTypeLike as NumpyDTypeLike


ArrayLike = TypeVar("ArrayLike", np.ndarray, torch.Tensor)
DTypeLike = TypeVar("DTypeLike", NumpyDTypeLike, torch.dtype)

ShapeType: TypeAlias = int | tuple[int, ...]
Number: TypeAlias = float | int
