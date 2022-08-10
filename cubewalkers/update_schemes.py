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
        Update mask array.
    """
    # much faster, but sometimes leads to wrong result if there are two maximums
    # z = cp.random.random((n,w), dtype=cp.float32)
    # z = z - cp.max(z, axis=0)
    # z = 1 + cp.sign(z)
    uinds = cp.random.randint(0, n, (w,))
    z = cp.zeros((n, w), dtype=cp.float32)
    z[uinds, cp.arange(w)] = 1
    return z

def asynchronous_PBN(t: int, n: int, w: int, a: cp.ndarray) -> cp.ndarray:
    """Update mask that randomly selects a single node to be updated at each timestep.
    Passes random values for PBN support. Each value is independently generated for each node.

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
    uinds = cp.random.randint(0, n, (w,))
    z = cp.zeros((n, w), dtype=cp.float32)
    z[uinds, cp.arange(w)] = 1
    x = 1 - cp.random.random(w, dtype=cp.float32)
    return x * z

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
    z = cp.random.random((n, w), dtype=cp.float32)
    z = cp.around(z)
    return z

def asynchronous_set_PBN(t: int, n: int, w: int, a: cp.ndarray) -> cp.ndarray:
    """Update mask that randomly selects a set of nodes to be updated at each timestep.
    Passes random values for PBN support. Each value is independently generated for each node.

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
    z = cp.random.random((n, w), dtype=cp.float32)
    z = cp.subtract(z, 0.5)
    z = cp.absolute(z) + z
    return z

def asynchronous_set_PBN_dependent(t: int, n: int, w: int, a: cp.ndarray) -> cp.ndarray:
    """Update mask that randomly selects a set of nodes to be updated at each timestep.
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
    a : cp.ndarray
        Current array of trajectories (not used)

    Returns
    -------
    cp.ndarray
        Update mask array. Entry value can be used by update function for PBN support.
    """
    z = cp.random.random((n, w), dtype=cp.float32)
    z = cp.around(z)
    x = 1 - cp.random.random(w, dtype=cp.float32)
    return x * z

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
    return 1 - cp.random.random((n, w), dtype=cp.float32)


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
    z = cp.ones((n, w), dtype=cp.float32)
    x = 1 - cp.random.random(w, dtype=cp.float32)
    return x * z
