"""This Module contains custom typing utilities for the `cubewalkers` package."""

from typing import TYPE_CHECKING, Any, Protocol

import cupy as cp  # type: ignore

if TYPE_CHECKING:

    class RawKernelType:
        """A dummy class for typing purposes.

        It represents a Python-callable CUDA
        kernel, such as can be generated using tools in the `parser` module.
        """

        def __call__(
            self,
            grid: tuple[int, ...],
            block: tuple[int, ...],
            args: tuple[Any, ...],
            **kwargs: Any,
        ):
            """_summary_

            Parameters
            ----------
            grid : tuple[int, ...]
                The number of blocks in each grid dimension.
            block : tuple[int, ...]
                The number of threads in each block dimension.
            args : tuple[Any, ...]
                An array of arguments to be passed to the kernel. Used in one of
                two ways, in the order presented here.
            input_array_to_update : cp.NDArray
                The N x W array of nodes values to be updated.
            update_scheme_mask : cp.NDArray
                The N x W array of update mask values, whose entries reflect
                probability of updating the corresponding node.
            output_array_after_update: cp.NDArray
                The N x W array of nodes values after update. This will be
                modified in-place.
            lookup_table : cp.NDArray, optional
                A  lookup table that contains the output column of each rule's
                update function. If provided, it is passed to the kernel, in
                which case the kernel must be a lookup-table-based kernel. If
                not provided, then the kernel must have the update rules
                internally encoded.
            current_time_step : int
                The current time step of the simulation. Used for time-dependent kernels.
            number_of_nodes : int
                The number of nodes in the network, N.
            number_of_walkers : int
                The number of walkers in the simulation, W.
            """
            return


if TYPE_CHECKING:

    class MaskFunctionType(Protocol):
        """
        A dummy class for typing purposes.

        It represents a function that
        generates the update mask arrays used in the `update_schemes` module.
        """

        def __call__(
            self, t: int, n: int, w: int, a: cp.typing.NDArray, **kwargs: Any
        ) -> cp.NDArray:
            """
            Called to generate the update mask array. Parameter may or may
            not be used, depending on the scheme.

            Parameters
            ----------
            t : int
                Current timestep value
            n : int
                Number of nodes
            w : int
                Number of ensemble walkers
            a : cp.NDArray
                Current array of trajectories

            Returns
            -------
            cp.NDArray
                The N x W array of update mask values, whose entries reflect
                probability of updating the corresponding node.
            """
            x: cp.NDArray = cp.array()  # type: ignore
            return x
