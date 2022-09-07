from __future__ import annotations

import cupy as cp


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import cana

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




