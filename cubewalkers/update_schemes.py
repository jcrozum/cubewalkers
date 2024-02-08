from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp  # type: ignore

if TYPE_CHECKING:
    from typing import Any

    from cubewalkers.custom_typing import RawKernelType

# Note: we are ignoring a lot of types in here because the type checker doesn't
# play nice with cupy

asynchronous_kernel: RawKernelType = cp.RawKernel(  # type: ignore
    r"""
extern "C" __global__
void asynchronous(const int* x1, int* z, int N, int W) {
    int w_reserved = blockDim.x * blockIdx.x + threadIdx.x;
    int n_reserved = blockDim.y * blockIdx.y + threadIdx.y;
    if(n_reserved < N && w_reserved < W) {
        if(x1[w_reserved]==n_reserved) {
            int a_reserved = n_reserved * W + w_reserved;
            z[a_reserved] = 1;
        }
    }
}
""",
    "asynchronous",
)


def asynchronous(t: int, n: int, w: int, a: cp.NDArray, **kwargs: Any) -> cp.NDArray:
    """
    Update mask that randomly selects a single node to be updated at each timestep.

    Parameters
    ----------
    t : int
        Current timestep value (not used)
    n : int
        Number of nodes
    w : int
        Number of ensemble walkers
    a : cp.NDArray
        Current array of trajectories (not used)

    Returns
    -------
    cp.NDArray
        Update mask array.
    """

    tpb = (32, 32)
    if "threads_per_block" in kwargs:
        tpb = kwargs["threads_per_block"]

    x1 = cp.floor(cp.random.random(size=(w,)) * n)  # type: ignore
    x1 = x1.astype(cp.int32)  # type: ignore
    z = cp.zeros((n, w), dtype=cp.int32)  # type: ignore
    asynchronous_kernel((w // tpb[1] + 1, n // tpb[0] + 1), tpb, (x1, z, n, w))  # type: ignore
    return z.astype(cp.float32)  # type: ignore


def asynchronous_PBN(
    t: int, n: int, w: int, a: cp.NDArray, **kwargs: Any
) -> cp.NDArray:
    """
    Update mask that randomly selects a single node to be updated at each timestep.
    Passes random values for PBN support. Each value is independently generated for each node.

    Parameters
    ----------
    t : int
        Current timestep value (not used)
    n : int
        Number of nodes
    w : int
        Number of ensemble walkers
    a : cp.NDArray
        Current array of trajectories (not used)

    Returns
    -------
    cp.NDArray
        Update mask array. Entry value can be used by update function for PBN support.
    """
    tpb = (32, 32)
    if "threads_per_block" in kwargs:
        tpb = kwargs["threads_per_block"]

    x1 = cp.floor(cp.random.random(size=(w,)) * n)  # type: ignore
    x1 = x1.astype(cp.int32)  # type: ignore
    z = cp.zeros((n, w), dtype=cp.int32)  # type: ignore
    asynchronous_kernel((w // tpb[1] + 1, n // tpb[0] + 1), tpb, (x1, z, n, w))  # type: ignore
    z = z.astype(cp.float32)  # type: ignore

    x = 1 - cp.random.random(w, dtype=cp.float32)  # type: ignore
    return x * z  # type: ignore


def asynchronous_set(
    t: int, n: int, w: int, a: cp.NDArray, **kwargs: Any
) -> cp.NDArray:
    """
    Update mask that randomly selects a set of nodes to be updated at each timestep.

    Parameters
    ----------
    t : int
        Current timestep value (not used)
    n : int
        Number of nodes
    w : int
        Number of ensemble walkers
    a : cp.NDArray
        Current array of trajectories (not used)

    Returns
    -------
    cp.NDArray
        Update mask array.
    """
    prob = 0.5
    if "set_update_prob" in kwargs:
        prob = kwargs["set_update_prob"]

    z = cp.random.random((n, w), dtype=cp.float32)  # type: ignore
    z = cp.ceil(prob - z)  # type: ignore
    return z  # type: ignore


def asynchronous_set_PBN(
    t: int, n: int, w: int, a: cp.NDArray, **kwargs: Any
) -> cp.NDArray:
    """
    Update mask that randomly selects a set of nodes to be updated at each timestep.
    Passes random values for PBN support. Each value is independently generated for each node.

    Parameters
    ----------
    t : int
        Current timestep value (not used)
    n : int
        Number of nodes
    w : int
        Number of ensemble walkers
    a : cp.NDArray
        Current array of trajectories (not used)

    Returns
    -------
    cp.NDArray
        Update mask array. Entry value can be used by update function for PBN support.
    """
    prob = 0.5
    if "set_update_prob" in kwargs:
        prob = kwargs["set_update_prob"]

    z = cp.random.random((n, w), dtype=cp.float32)  # type: ignore
    mz = cp.ceil(prob - z)  # type: ignore
    return z * mz / prob  # type: ignore


def asynchronous_set_PBN_dependent(
    t: int, n: int, w: int, a: cp.NDArray, **kwargs: Any
) -> cp.NDArray:
    """
    Update mask that randomly selects a set of nodes to be updated at each timestep.
    Passes random values for PBN support. All nodes use the same value,
    but each walker uses an independently generated value.

    Parameters
    ----------
    t : int
        Current timestep value (not used)
    n : int
        Number of nodes
    w : int
        Number of ensemble walkers
    a : cp.NDArray
        Current array of trajectories (not used)

    Returns
    -------
    cp.NDArray
        Update mask array. Entry value can be used by update function for PBN support.
    """
    prob = 0.5
    if "set_update_prob" in kwargs:
        prob = kwargs["set_update_prob"]

    z = cp.random.random((n, w), dtype=cp.float32)  # type: ignore
    z = cp.ceil(prob - z)  # type: ignore
    x = 1 - cp.random.random(w, dtype=cp.float32)  # type: ignore
    return x * z  # type: ignore


def synchronous(t: int, n: int, w: int, a: cp.NDArray, **kwargs: Any) -> cp.NDArray:
    """
    Update mask that updates all nodes at each timestep.

    Parameters
    ----------
    t : int
        Current timestep value (not used)
    n : int
        Number of nodes
    w : int
        Number of ensemble walkers
    a : cp.NDArray
        Current array of trajectories (not used)

    Returns
    -------
    cp.NDArray
        Update mask array. Entry value can be used by update function for PBN support.
    """
    return cp.ones((n, w), dtype=cp.float32)  # type: ignore


def synchronous_PBN(t: int, n: int, w: int, a: cp.NDArray, **kwargs: Any) -> cp.NDArray:
    """
    Update mask that updates all nodes at each timestep. Passes random values for PBN
    support. Each value is indepently generated for each node.

    Parameters
    ----------
    t : int
        Current timestep value (not used)
    n : int
        Number of nodes
    w : int
        Number of ensemble walkers
    a : cp.NDArray
        Current array of trajectories (not used)

    Returns
    -------
    cp.NDArray
        Update mask array. Entry value can be used by update function for PBN support.
    """
    return 1 - cp.random.random((n, w), dtype=cp.float32)  # type: ignore


def synchronous_PBN_dependent(
    t: int, n: int, w: int, a: cp.NDArray, **kwargs: Any
) -> cp.NDArray:
    """
    Update mask that updates all nodes at each timestep. Passes random values for PBN
    support. All nodes use the same value, but each walker uses an independently generated
    value.

    Parameters
    ----------
    t : int
        Current timestep value (not used)
    n : int
        Number of nodes
    w : int
        Number of ensemble walkers
    a : cp.NDArray
        Current array of trajectories (not used)

    Returns
    -------
    cp.NDArray
        Update mask array. Entry value can be used by update function for PBN support.
    """
    z = cp.ones((n, w), dtype=cp.float32)  # type: ignore
    x = 1 - cp.random.random(w, dtype=cp.float32)  # type: ignore
    return x * z  # type: ignore
