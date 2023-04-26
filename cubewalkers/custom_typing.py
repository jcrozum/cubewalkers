from typing import Any, Callable
import cupy as cp  # type: ignore


RawKernelType = Any  # TypeVar('RawKernelType')
MaskFunctionType = Callable[[
    int, int, int, cp.typing.NDArray], cp.typing.NDArray]
