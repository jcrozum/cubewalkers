from typing import TYPE_CHECKING, Any, Protocol

import cupy as cp  # type: ignore

if TYPE_CHECKING:

    class RawKernelType:
        def __call__(
            self,
            grid: tuple[int, ...],
            block: tuple[int, ...],
            args: tuple[Any, ...],
            **kwargs: Any,
        ):
            return

    class MaskFunctionType(Protocol):
        def __call__(
            self, t: int, n: int, w: int, a: cp.typing.NDArray, **kwargs: Any
        ) -> cp.NDArray:
            x: cp.NDArray = cp.array()  # type: ignore
            return x
