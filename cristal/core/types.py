from typing import TypeAlias, TypeVar

import numpy as np
import torch

ArrayLike = TypeVar("ArrayLike", np.ndarray, torch.Tensor)
DTypeLike = TypeVar("DTypeLike", np.dtype, torch.dtype, type[int], type[float])

ShapeType: TypeAlias = tuple[int, ...]
Number: TypeAlias = float | int
