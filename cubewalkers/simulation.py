from __future__ import annotations
import cupy as cp
import cubewalkers.update_schemes as cw_update_schemes


def simulate_random_ensemble(kernel: cp.RawKernel,
                             N: int, T: int, W: int,
                             averages_only: bool = False,
                             maskfunction: callable | None = None,
                             threads_per_block: tuple[int, int] = (32, 32)) -> cp.array:
    # initialize output array (will copy to input on first timestep)
    out = cp.random.choice([cp.bool_(0), cp.bool_(1)], (N, W))

    # compute blocks per grid based on number of walkers & variables and threads_per_block
    blocks_per_grid = (out.shape[1] // threads_per_block[1]+1,
                       out.shape[0] // threads_per_block[0]+1)

    # set updating scheme
    if maskfunction is None or maskfunction == 'asynchronous':
        def maskfunction(t, n, w, a):
            return cw_update_schemes.general_asynchronous_update_mask(t, n, w, a)
    elif maskfunction == 'fully_asynchronous':
        def maskfunction(t, n, w, a):
            return cw_update_schemes.fully_asynchronous_update_mask(t, n, w, a)
    elif maskfunction == 'synchronous':
        def maskfunction(t, n, w, a):
            return cw_update_schemes.synchronous_update_mask(t, n, w, a)
    elif maskfunction == 'synchronous_PBN':
        def maskfunction(t, n, w, a):
            return cw_update_schemes.synchronous_update_mask_PBN(t, n, w, a)
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
