from typing import Iterator, Literal, get_args

from ..core.types import ArrayLike

IMPLEMENTED_STORAGE = Literal["full", "batch"]


class Storage:
    def __init__(self, method: IMPLEMENTED_STORAGE = "batch", batch_size: int = 256) -> None:
        assert method in get_args(IMPLEMENTED_STORAGE), f"method must be in {IMPLEMENTED_STORAGE}. Got {method}."
        assert batch_size > 0, f"batch_size must be positive. Got {batch_size}."

        self.method = method
        self.batch_size = batch_size

    def iterate(self, X: ArrayLike) -> Iterator[ArrayLike]:
        if self.method == "full":
            yield X
        elif self.method == "batch":
            if self.batch_size <= 0:
                raise ValueError("Batch size must be positive")
            for i in range(0, len(X), self.batch_size):
                yield X[i : i + self.batch_size]
        else:
            raise ValueError("Method must be either 'full' or 'batch'")

    def __call__(self, X: ArrayLike) -> Iterator[ArrayLike]:
        return self.iterate(X)
