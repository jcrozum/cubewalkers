from __future__ import annotations
import cupy as cp
from cubewalkers.update_schemes import synchronous


def simulate_ensemble(kernel: cp.RawKernel,
                      N: int, T: int, W: int,
                      averages_only: bool = False,
                      initial_states: cp.ndarray | None = None,
                      maskfunction: callable = synchronous,
                      threads_per_block: tuple[int, int] = (32, 32)) -> cp.ndarray:
    """Simulates a random ensemble of walkers on a Boolean network using the input kernel.

    Parameters
    ----------
    kernel : cp.RawKernel
        CuPy RawKernel that provides the update functions (see parser module).
    N : int
        Number of nodes in the network.
    T : int
        Number of timesteps to simulate.
    W : int
        Number of ensemble walkers to simulate.
    initial_states : cp.ndarray | None, optional
        N x W array of initial states. Must be a cupy ndarray of type cupy.bool_. If None
        (default), initial states are randomly initialized.
    averages_only : bool, optional
        If True, stores only average node values at each timestep. 
        Otherwise, stores node values for each walker. By default False.
    maskfunction : callable, optional
        Function that returns a mask for selecting which node values to update. 
        By default, uses the synchronous update scheme. See update_schemes for examples.
    threads_per_block : tuple[int, int], optional
        How many threads should be in each block for each dimension of the N x W array, 
        by default (32, 32). See CUDA documentation for details.

    Returns
    -------
    cp.ndarray
        If averages_only is False (default), a T x N x W array of node values at each timestep
        for each node and walker. If averages_only is True, a T x N array of average node
        values.
    """
    # initialize output array (will copy to input on first timestep)
    if initial_states is None:
        out = cp.random.choice([cp.bool_(0), cp.bool_(1)], (N, W))
    else:
        out = initial_states.copy()

    # compute blocks per grid based on number of walkers & variables and threads_per_block
    blocks_per_grid = (out.shape[1] // threads_per_block[1]+1,
                       out.shape[0] // threads_per_block[0]+1)

    # initialize return array
    if averages_only:
        trajectories = cp.ones((T+1, N))
        trajectories[0] = cp.mean(out, axis=1)
    else:
        trajectories = cp.ones((T+1, N, W))
        trajectories[0, :, :] = out.copy()

    # simulation begins here
    for t in range(T):
        arr = out.copy()  # get values from update

        # compute which variables to update
        mask = maskfunction(t, N, W, arr)

        # run the update on the GPU
        kernel(blocks_per_grid, threads_per_block, (arr, mask, out, t, N, W))

        # store results
        if averages_only:
            trajectories[t+1] = cp.mean(out, axis=1)
        else:
            trajectories[t+1, :, :] = out.copy()

    return trajectories
