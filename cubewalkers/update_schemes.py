from __future__ import annotations
import cupy as cp


def asynchronous(t: int, n: int, w: int, a: cp.ndarray) -> cp.ndarray:
    """Update mask that randomly selects a single node to be updated at each timestep.

    Parameters
    ----------
    t : int
        Current timestep value (not used)
    n : int
        Number of nodes
    w : int
        Number of ensemble walkers
    a : cp.ndarray
        Current array of trajectories (not used)

    Returns
    -------
    cp.ndarray
        Update mask array. Entry value can be used by update function for PBN support.
    """
    uinds = cp.random.randint(0, w, (n,))
    z = cp.zeros((n, w), dtype=cp.float32)
    z[cp.arange(n), uinds] = cp.float32(1)
    return z


def asynchronous_set(t: int, n: int, w: int, a: cp.ndarray) -> cp.ndarray:
    """Update mask that randomly selects a set of nodes to be updated at each timestep.

    Parameters
    ----------
    t : int
        Current timestep value (not used)
    n : int
        Number of nodes
    w : int
        Number of ensemble walkers
    a : cp.ndarray
        Current array of trajectories (not used)

    Returns
    -------
    cp.ndarray
        Update mask array.
    """
    return cp.random.choice([cp.bool_(0), cp.bool_(1)], (n, w))


def synchronous(t: int, n: int, w: int, a: cp.ndarray) -> cp.ndarray:
    """Update mask that updates all nodes at each timestep.

    Parameters
    ----------
    t : int
        Current timestep value (not used)
    n : int
        Number of nodes
    w : int
        Number of ensemble walkers
    a : cp.ndarray
        Current array of trajectories (not used)

    Returns
    -------
    cp.ndarray
        Update mask array. Entry value can be used by update function for PBN support.
    """
    return cp.ones((n, w), dtype=cp.float32)


def synchronous_PBN(t: int, n: int, w: int, a: cp.ndarray) -> cp.ndarray:
    """Update mask that updates all nodes at each timestep. Passes random values for PBN
    support. Each value is indepently generated for each node.

    Parameters
    ----------
    t : int
        Current timestep value (not used)
    n : int
        Number of nodes
    w : int
        Number of ensemble walkers
    a : cp.ndarray
        Current array of trajectories (not used)

    Returns
    -------
    cp.ndarray
        Update mask array. Entry value can be used by update function for PBN support.
    """
    return cp.random.random((n, w), dtype=cp.float32)


def synchronous_PBN_dependent(t: int, n: int, w: int, a: cp.ndarray) -> cp.ndarray:
    """Update mask that updates all nodes at each timestep. Passes random values for PBN
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
    a : cp.ndarray
        Current array of trajectories (not used)

    Returns
    -------
    cp.ndarray
        Update mask array. Entry value can be used by update function for PBN support.
    """
    return cp.random.random(w, dtype=cp.float32) * cp.ones((n, w), dtype=cp.float32)
