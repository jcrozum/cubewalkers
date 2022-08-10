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
    # compute blocks per grid based on number of walkers & variables and threads_per_block
    blocks_per_grid = (W // threads_per_block[1]+1,
                       N // threads_per_block[0]+1)

    # initialize output array (will copy to input on first timestep)
    if initial_states is None:
        out = cp.random.choice([cp.bool_(0), cp.bool_(1)], (N, W))
    else:
        out = initial_states.copy()

    # initialize return array
    if averages_only:
        trajectories = -cp.ones((T+1, N))
        trajectories[0] = cp.mean(out, axis=1)
    else:
        trajectories = -cp.ones((T+1, N, W))
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


def dynamical_impact(kernel: cp.RawKernel, source: int,
                     N: int, T: int, W: int,
                     maskfunction: callable = synchronous,
                     threads_per_block: tuple[int, int] = (32, 32)) -> cp.ndarray:
    """Computes the dynamical impact of the source node index on all others (including
    itself, from time=0 to time=T).

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
    maskfunction : callable, optional
        Function that returns a mask for selecting which node values to update. 
        By default, uses the synchronous update scheme. See update_schemes for examples.
        For dynamical impact, if the maskfunction is state-dependent, then the unperturbed
        trajectory is used.
    threads_per_block : tuple[int, int], optional
        How many threads should be in each block for each dimension of the N x W array, 
        by default (32, 32). See CUDA documentation for details.

    Returns
    -------
    cp.ndarray
        (T+1) x N array of dynamical impacts of the source at each time.
    """
    # compute blocks per grid based on number of walkers & variables and threads_per_block
    blocks_per_grid = (W // threads_per_block[1]+1,
                       N // threads_per_block[0]+1)

    # initial conditions
    outU = cp.random.choice([cp.bool_(0), cp.bool_(1)], (N, W))
    outP = outU.copy()
    outP[source, :] = ~outP[source, :]

    # initialize impact array
    impact = -cp.ones((T+1, N))
    # compute impact[0] = cp.mean(outU ^ outP, axis = 1), but more efficiently 
    # beause we know outU and outP only differ in the value of the source initially
    impact[0] = cp.zeros((N,))
    impact[0][source] = 1.
    
    # begin simulation
    for t in range(T):
        arrU = outU.copy()  # get values from update
        arrP = outP.copy()
        
        mask = maskfunction(t, N, W, arrU)
        
        # run the update on the GPU for the two states
        kernel(blocks_per_grid, threads_per_block, (arrU, mask, outU, t, N, W))
        kernel(blocks_per_grid, threads_per_block, (arrP, mask, outP, t, N, W))
        
        impact[t+1] = cp.mean(outU ^ outP, axis = 1)
    
    return impact