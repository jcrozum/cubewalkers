from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp  # type: ignore

from cubewalkers.update_schemes import asynchronous, synchronous

if TYPE_CHECKING:
    from cubewalkers.custom_typing import MaskFunctionType, RawKernelType


def simulate_ensemble(
    kernel: RawKernelType,
    N: int,
    T: int,
    W: int,
    T_window: int | None = None,
    lookup_tables: cp.NDArray | None = None,
    averages_only: bool = False,
    initial_states: cp.NDArray | None = None,
    maskfunction: MaskFunctionType = synchronous,
    threads_per_block: tuple[int, int] = (32, 32),
    set_update_prob: float = 0.5,
) -> cp.NDArray:
    """
    Simulates a random ensemble of walkers on a Boolean network using the input kernel.

    Parameters
    ----------
    kernel : RawKernelType
        CuPy RawKernel that provides the update functions (see parser module).
    N : int
        Number of nodes in the network.
    T : int
        Number of timesteps to simulate.
    W : int
        Number of ensemble walkers to simulate.
    T_window : int, optional
        Number of time points to keep (from t=T-T_window+1 to t=T). If None (default),
        keep all time points.
    lookup_tables : cp.NDArray, optional
        A merged lookup table that contains the output column of each rule's
        lookup table (padded by False values). If provided, it is passed to the kernel,
        in which case the kernel must be a lookup-table-based kernel. If None (default),
        then the kernel must have the update rules internally encoded.
    initial_states : cp.NDArray | None, optional
        N x W array of initial states. Must be a CuPy ndarray of CuPy Boolean values. If None
        (default), initial states are randomly initialized.
    averages_only : bool, optional
        If True, stores only average node values at each timestep.
        Otherwise, stores node values for each walker. By default False.
    maskfunction : MaskFunctionType, optional
        Function that returns a mask for selecting which node values to update.
        By default, uses the synchronous update scheme. See update_schemes for examples.
    threads_per_block : tuple[int, int], optional
        How many threads should be in each block for each dimension of the N x W array,
        by default (32, 32). See CUDA documentation for details.

    Returns
    -------
    cp.NDArray
        If averages_only is False (default), a T x N x W array of node values at each timestep
        for each node and walker. If averages_only is True, a T x N array of average node
        values.
    """
    # compute blocks per grid based on number of walkers & variables and threads_per_block
    blocks_per_grid = (W // threads_per_block[1] + 1, N // threads_per_block[0] + 1)

    # If we're using lookup tables for the rule evaluation, record the size of the largest
    # table (note that the LUTs must be padded to equal lengths)
    if lookup_tables is not None:
        table_length = len(lookup_tables[0])
    else:
        table_length = None

    # initialize output array (will copy to input on first timestep)
    if initial_states is None:
        out: cp.NDArray = cp.random.choice([cp.bool_(0), cp.bool_(1)], (N, W))  # type: ignore
    else:
        out = initial_states.copy()

    if T_window is None or T_window > T or T_window < 1:
        T_window = T + 1

    # initialize return array
    if averages_only:
        trajectories = cp.ones((T_window, N), dtype=cp.float_)  # type: ignore
        trajectories[0] = cp.mean(out, axis=1)  # type: ignore
    else:
        trajectories = cp.ones((T_window, N, W), dtype=cp.bool_)  # type: ignore
        trajectories[0, :, :] = out.copy()

    # simulation begins here
    for t in range(T):
        arr = out.copy()  # get values from update

        # compute which variables to update
        mask = maskfunction(
            t,
            N,
            W,
            arr,
            threads_per_block=threads_per_block,
            set_update_prob=set_update_prob,
        )

        # run the update on the GPU
        if table_length is None:
            kernel(blocks_per_grid, threads_per_block, (arr, mask, out, t, N, W))
        else:
            kernel(
                blocks_per_grid,
                threads_per_block,
                (arr, mask, out, lookup_tables, t, N, W, table_length),
            )

        # store results
        # t_ind = (t+1) % T_window
        if t >= T - T_window:
            if averages_only:
                trajectories[t - (T - T_window)] = cp.mean(out, axis=1)  # type: ignore
            else:
                trajectories[t - (T - T_window), :, :] = out.copy()

    # if t_ind > 0:
    #     trajectories = cp.concatenate(
    #         (trajectories[t_ind+1:], trajectories[0:t_ind+1]), axis=0)

    return trajectories  # type: ignore


def simulate_perturbation(
    kernel: RawKernelType,
    source: int | list[int],
    N: int,
    T: int,
    W: int,
    T_sample: int = 1,
    lookup_tables: cp.NDArray | None = None,
    maskfunction: MaskFunctionType = synchronous,
    threads_per_block: tuple[int, int] = (32, 32),
) -> tuple[cp.NDArray, cp.NDArray, cp.NDArray]:
    """
    Computes the trajectories in response to a perturbation of the source node index,
    and returns summed up trajectories and summed up differences (3 arrays of N*W)

    Parameters
    ----------
    kernel : RawKernelType
        CuPy RawKernel that provides the update functions (see parser module).
    source : int | list[int]
        Index or indices of node(s) to perturb.
    N : int
        Number of nodes in the network.
    T : int
        Number of timesteps to simulate.
    W : int
        Number of ensemble walkers to simulate.
    T_sample : int, optional
        Number of time points to use for summing (t=T-T_sample+1 to t=T), by default, 1.
    lookup_tables : cp.NDArray, optional
        A merged lookup table that contains the output column of each rule's
        lookup table (padded by False values). If provided, it is passed to the kernel,
        in which case the kernel must be a lookup-table-based kernel. If None (default),
        then the kernel must have the update rules internally encoded.
    maskfunction : MaskFunctionType, optional
        Function that returns a mask for selecting which node values to update.
        By default, uses the synchronous update scheme. See update_schemes for examples.
        If the maskfunction is state-dependent, then the unperturbed trajectory is used.
    threads_per_block : tuple[int, int], optional
        How many threads should be in each block for each dimension of the N x W array,
        by default (32, 32). See CUDA documentation for details.

    Returns
    -------
    trajU : cp.NDArray
        The trajectory without perturbation summed up from t=T-T_sample+1 to t=T.
    trajP : cp.NDArray
        The trajectories in response to the source node perturbation summed up from t=T-T_sample+1 to t=T.
    diff : cp.NDArray
        The differences in trajU and trajP summed up from t=T-T_sample+1 to t=T.
    """
    # compute blocks per grid based on number of walkers & variables and threads_per_block
    blocks_per_grid = (W // threads_per_block[1] + 1, N // threads_per_block[0] + 1)

    if T_sample < 1 or T_sample > T:
        T_sample = 1

    # If we're using lookup tables for the rule evaluation, record the size of the largest
    # table (note that the LUTs must be padded to equal lengths)
    if lookup_tables is not None:
        table_length = len(lookup_tables[0])
    else:
        table_length = None

    # initial conditions
    outU: cp.NDArray = cp.random.choice([cp.bool_(0), cp.bool_(1)], (N, W))  # type: ignore
    outP = outU.copy()
    outP[source, :] = ~outP[source, :]

    # store trajectories for quasicoherence computation
    trajU: cp.NDArray = cp.zeros((N, W), dtype=cp.int32)  # type: ignore
    trajP: cp.NDArray = cp.zeros((N, W), dtype=cp.int32)  # type: ignore
    diff: cp.NDArray = cp.zeros((N, W), dtype=cp.int32)  # type: ignore

    # begin simulation
    for t in range(T):
        arrU = outU.copy()  # get values from update
        arrP = outP.copy()

        mask = maskfunction(t, N, W, arrU)

        # run the update on the GPU for the two states
        if table_length is None:
            kernel(blocks_per_grid, threads_per_block, (arrU, mask, outU, t, N, W))
            kernel(blocks_per_grid, threads_per_block, (arrP, mask, outP, t, N, W))
        else:
            kernel(
                blocks_per_grid,
                threads_per_block,
                (arrU, mask, outU, lookup_tables, t, N, W, table_length),
            )
            kernel(
                blocks_per_grid,
                threads_per_block,
                (arrP, mask, outP, lookup_tables, t, N, W, table_length),
            )
        if t >= T - T_sample:
            trajU[:, :] += outU.astype(cp.int32)
            trajP[:, :] += outP.astype(cp.int32)
            diff[:, :] += (outU ^ outP).astype(cp.int32)

    return trajU, trajP, diff


def source_quasicoherence(
    trajU: cp.NDArray,
    trajP: cp.NDArray,
    T_sample: int = 1,
    fuzzy_coherence: bool = False,
) -> cp.NDArray:
    """
    Computes the quasicoherence in response to perturbation of source node index,
    averaging trajectories from t=T-T_sample+1 to T.

    Parameters
    ----------
    trajU, trajP : cp.NDArray
        The trajectories in response to the source node perturbation summed up from t=T-T_sample+1 to t=T.
    T_sample : int, optional
        Number of time points to use for averaging (t=T-T_sample+1 to t=T), by default, 1.
    fuzzy_coherence : bool, optional
        If False (default), trajectroies are marked as either in agreement (1) or not in
        agreement (0) depending on whether fixed nodes are in agreement. If True, the
        average absolute difference between state vectors is used instead.

    Returns
    -------
    cp.NDArray
        The estimated value of the quasicoherence response to the source node perturbation.
    """

    if fuzzy_coherence:
        quasicoherence_array: cp.NDArray = 1 - cp.abs(trajU - trajP) / T_sample  # type: ignore
        quasicoherence: cp.NDArray = cp.mean(quasicoherence_array)  # type: ignore
    else:
        quasicoherence_array = (
            (trajU == T_sample) & (trajP == T_sample)
            | (trajU == 0) & (trajP == 0)
            | (trajU > 0) & (trajP > 0) & (trajU < T_sample) & (trajP < T_sample)
        )

        quasicoherence = cp.mean(cp.mean(quasicoherence_array, axis=0) == 1)  # type: ignore

    return quasicoherence


def source_final_hamming_distance(diff: cp.NDArray, T_sample: int = 1) -> cp.NDArray:
    """
    Computes the final hamming distance in response to perturbation of source node index,
    averaging hamming distances from t=T-T_sample+1 to T.

    Parameters
    ----------
    diff : cp.NDArray
        The trajectories in response to the source node perturbation.
    T_sample : int, optional
        Number of time points to use for averaging (t=T-T_sample+1 to t=T), by default, 1.

    Returns
    -------
    cp.NDArray
        The estimated value of the final hamming distance response to the source node perturbation.
    """

    final_hamming_distance_array = diff / T_sample
    final_hamming_distance: cp.NDArray = cp.mean(  # type: ignore
        cp.sum(final_hamming_distance_array, axis=0)  # type: ignore
    )

    return final_hamming_distance


def dynamical_impact(
    kernel: RawKernelType,
    source: int | list[int],
    N: int,
    T: int,
    W: int,
    lookup_tables: cp.NDArray | None = None,
    maskfunction: MaskFunctionType = synchronous,
    threads_per_block: tuple[int, int] = (32, 32),
) -> cp.NDArray:
    """
    Computes the dynamical impact of the source node index on all others (including
    itself, from time=0 to time=T).

    Parameters
    ----------
    kernel : RawKernelType
        CuPy RawKernel that provides the update functions (see parser module).
    source : int | list[int]
        Index or indices of node(s) to perturb for dynamical impact calculation.
    N : int
        Number of nodes in the network.
    T : int
        Number of timesteps to simulate.
    W : int
        Number of ensemble walkers to simulate.
    lookup_tables : cp.NDArray, optional
        A merged lookup table that contains the output column of each rule's
        lookup table (padded by False values). If provided, it is passed to the kernel,
        in which case the kernel must be a lookup-table-based kernel. If None (default),
        then the kernel must have the update rules internally encoded.
    maskfunction : MaskFunctionType, optional
        Function that returns a mask for selecting which node values to update.
        By default, uses the synchronous update scheme. See update_schemes for examples.
        For dynamical impact, if the maskfunction is state-dependent, then the unperturbed
        trajectory is used.
    threads_per_block : tuple[int, int], optional
        How many threads should be in each block for each dimension of the N x W array,
        by default (32, 32). See CUDA documentation for details.

    Returns
    -------
    cp.NDArray
        (T+1) x N array of dynamical impacts of the source at each time.
    """
    # compute blocks per grid based on number of walkers & variables and threads_per_block
    blocks_per_grid = (W // threads_per_block[1] + 1, N // threads_per_block[0] + 1)

    # If we're using lookup tables for the rule evaluation, record the size of the largest
    # table (note that the LUTs must be padded to equal lengths)
    if lookup_tables is not None:
        table_length = len(lookup_tables[0])
    else:
        table_length = None

    # initial conditions
    outU: cp.NDArray = cp.random.choice([cp.bool_(0), cp.bool_(1)], (N, W))  # type: ignore
    outP = outU.copy()
    outP[source, :] = ~outP[source, :]

    # initialize impact array
    impact: cp.NDArray = -cp.ones((T + 1, N))  # type: ignore
    # compute impact[0] = cp.mean(outU ^ outP, axis = 1), but more efficiently
    # beause we know outU and outP only differ in the value of the source initially
    impact[0] = cp.zeros((N,))  # type: ignore
    impact[0][source] = 1.0

    # begin simulation
    for t in range(T):
        arrU = outU.copy()  # get values from update
        arrP = outP.copy()

        mask = maskfunction(t, N, W, arrU)

        # run the update on the GPU for the two states
        if table_length is None:
            kernel(blocks_per_grid, threads_per_block, (arrU, mask, outU, t, N, W))
            kernel(blocks_per_grid, threads_per_block, (arrP, mask, outP, t, N, W))
        else:
            kernel(
                blocks_per_grid,
                threads_per_block,
                (arrU, mask, outU, lookup_tables, t, N, W, table_length),
            )
            kernel(
                blocks_per_grid,
                threads_per_block,
                (arrP, mask, outP, lookup_tables, t, N, W, table_length),
            )

        impact[t + 1] = cp.mean(outU ^ outP, axis=1)  # type: ignore

    return impact


def derrida_coefficient(
    kernel: RawKernelType,
    N: int,
    W: int,
    lookup_tables: cp.NDArray | None = None,
    threads_per_block: tuple[int, int] = (32, 32),
) -> float:
    """
    Estimates the Derrida coefficient.

    Parameters
    ----------
    kernel : RawKernelType
        CuPy RawKernel that provides the update functions (see parser module).
    N : int
        Number of nodes in the network.
    W : int
        Number of ensemble walkers to simulate.
    lookup_tables : cp.NDArray, optional
        A merged lookup table that contains the output column of each rule's
        lookup table (padded by False values). If provided, it is passed to the kernel,
        in which case the kernel must be a lookup-table-based kernel. If None (default),
        then the kernel must have the update rules internally encoded.
    threads_per_block : tuple[int, int], optional
        How many threads should be in each block for each dimension of the N x W array,
        by default (32, 32). See CUDA documentation for details.

    Returns
    -------
    float
        Derrida coefficient
    """

    # compute blocks per grid based on number of walkers & variables and threads_per_block
    blocks_per_grid = (W // threads_per_block[1] + 1, N // threads_per_block[0] + 1)

    if lookup_tables is not None:
        table_length = lookup_tables.shape[1]
    else:
        table_length = None

    # initial conditions
    outU: cp.NDArray = cp.random.random((N, W), dtype=cp.float32)  # type: ignore
    outU = cp.ceil(0.5 - outU).astype(cp.bool_)  # type: ignore
    outP = outU ^ asynchronous(0, N, W, None).astype(cp.bool_)

    # get values from update
    arrU = outU.copy()  # get values from update
    arrP = outP.copy()

    mask = synchronous(0, N, W, None)  # only defined for synchronous update

    # run the update on the GPU for the two states
    if table_length is None:
        kernel(blocks_per_grid, threads_per_block, (arrU, mask, outU, 1, N, W))
        kernel(blocks_per_grid, threads_per_block, (arrP, mask, outP, 1, N, W))
    else:
        kernel(
            blocks_per_grid,
            threads_per_block,
            (arrU, mask, outU, lookup_tables, 1, N, W, table_length),
        )
        kernel(
            blocks_per_grid,
            threads_per_block,
            (arrP, mask, outP, lookup_tables, 1, N, W, table_length),
        )

    return cp.mean(outU ^ outP) * N  # type: ignore
