from typing import Iterator, Literal, get_args

from ..types import ArrayLike

IMPLEMENTED_STORAGE = Literal["full", "batch"]


class Storage:
    def __init__(self, method: IMPLEMENTED_STORAGE = "batch", batch_size: int = 256) -> None:
        if method not in get_args(IMPLEMENTED_STORAGE):
            raise ValueError(f"method must be in {IMPLEMENTED_STORAGE}. Got {method}.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive. Got {batch_size}.")

        self.method = method
        self.batch_size = batch_size

    def iterate(self, X: ArrayLike) -> Iterator[ArrayLike]:
        # Full method
        if self.method == "full":
            yield X

        # Batch method
        else:
            for i in range(0, len(X), self.batch_size):
                yield X[i : i + self.batch_size]

    def __call__(self, X: ArrayLike) -> Iterator[ArrayLike]:
        yield from self.iterate(X)
