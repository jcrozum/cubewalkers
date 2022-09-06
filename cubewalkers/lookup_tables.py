from __future__ import annotations
from cubewalkers.update_schemes import synchronous
import cupy as cp


from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    import cana

# TODO: move this into the CANA import module
# TODO: make test for this function
def cana2cupyLUT(net: cana.BooleanNetwork) -> tuple[cp.ndarray, cp.ndarray]:
    """Extract lookup tables and input lists from a CANA network into a CuPy-compatible form.

    Parameters
    ----------
    net : cana.BooleanNetwork
        CANA network to import.

    Returns
    -------
    tuple[cp.ndarray, cp.ndarray]
        Returns a merged lookup table that contains the output column of each rule's
        lookup table (padded by False values). Also returns and inputs table, which 
        contains the inputs for each node (padded by -1).
    """
    outs = [u.outputs for u in net.nodes]
    outmax = len(max(outs, key=lambda x: len(x)))
    outs_bool = [list(map(lambda x: x == '1', out))  # convert to bool
                 + [False]*(outmax-len(out)) for out in outs]  # and pad for cupy

    inps = [u.inputs for u in net.nodes]
    inpmax = len(max(inps, key=lambda x: len(x)))
    inps_pad = [inp  # convert to bool
                + [-1]*(inpmax-len(inp)) for inp in inps]  # pad for cupy

    return cp.array(outs_bool, dtype=cp.bool_), cp.array(inps_pad, dtype=cp.int32)

# TODO: integrate with the Model class (or a new type of class)
def lut_kernel(inputs: Iterable[Iterable[int]], kernel_name: str) -> cp.RawKernel:
    """Constructs a CuPy RawKernel that simulates a network using lookup tables.

    Parameters
    ----------
    inputs : Iterable[Iterable[int]]
        Iterable i should contain the indicies of the nodes that regulate node i, 
        optionally padded by negative values.
    kernel_name : str
        Name of the kernel to be generated.

    Returns
    -------
    cp.RawKernel
        A CuPy RawKernel that accepts arguments in the following fashion:
        kernel(blocks_per_grid, threads_per_block, (
            input_array_to_update, 
            update_scheme_mask, 
            output_array_after_update, 
            lookup_table,
            current_time_step, 
            number_of_nodes, 
            number_of_walkers))
        and modifies the output_array_after_update in-place using the provided lookup 
        table to compute state transitions.
    """
    cpp_body = (
        f'extern "C" __global__\n'
        f'void {kernel_name}(const bool* A__reserved_input,\n'
        f'        const float* A__reserved_mask,\n'
        f'        bool* A__reserved_output,\n'
        f'        const bool* A__reserved_LUT,\n'
        f'        int t__reserved, int N__reserved, int W__reserved, int L__reserved) {{\n'
        f'    int w__reserved = blockDim.x * blockIdx.x + threadIdx.x;\n'
        f'    int n__reserved = blockDim.y * blockIdx.y + threadIdx.y;\n'
        f'    int a__reserved = w__reserved + n__reserved*W__reserved;\n'
        f'    if(n__reserved < N__reserved && w__reserved < W__reserved){{\n'
        f'        if(A__reserved_mask[a__reserved]>0){{\n'
    )

    for ln,input in enumerate(inputs):
        lookup_index = '0'
        for ind in input:
            if ind < 0:
                break
            lookup_index = (
                f'({lookup_index}<<1)'
                f'+A__reserved_input[{ind}*W__reserved+w__reserved]'
            )

        cpp_body += (
            f'if (n__reserved=={ln}){{'
            f'A__reserved_output[a__reserved]='
            f'A__reserved_LUT[({lookup_index})+L__reserved*n__reserved];}}\n'
        )
    cpp_body += '\n} else{A__reserved_output[a__reserved]=A__reserved_input[a__reserved];}}}'
    
    #print(cpp_body)
    return cp.RawKernel(cpp_body, kernel_name)

# TODO: integrate with the Model class (or a new type of class)
# TODO: make test for this function
def simulate_lut_kernel(kernel: cp.RawKernel,
                        lookup_tables: cp.ndarray,
                        N: int, T: int, W: int, L: int,
                        averages_only: bool = False,
                        initial_states: cp.ndarray | None = None,
                        maskfunction: callable = synchronous,
                        threads_per_block: tuple[int, int] = (32, 32)) -> cp.ndarray:
    """_summary_

    Parameters
    ----------
    kernel : cp.RawKernel
        CuPy RawKernel that provides the update functions as lookup table rules 
        (see lut_kernel function).
    lookup_tables : cp.ndarray
        A merged lookup table that contains the output column of each rule's
        lookup table (padded by False values)
    N : int
        Number of nodes in the network.
    T : int
        Number of timesteps to simulate.
    W : int
        Number of ensemble walkers to simulate.
    L : int
        The number of entries in the largest node lookup table in the network
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
    kernel : cp.RawKernel
        _description_
    
    Returns
    -------
    cp.ndarray
        _description_
    """
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
        kernel(blocks_per_grid, threads_per_block,
               (arr, mask, out, lookup_tables, t, N, W, L))

        # store results
        if averages_only:
            trajectories[t+1] = cp.mean(out, axis=1)
        else:
            trajectories[t+1, :, :] = out.copy()

    return trajectories



