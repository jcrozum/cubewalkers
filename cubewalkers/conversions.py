from __future__ import annotations

import cupy as cp
from cubewalkers import parser

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


def node_rule_from_cana(node: cana.BooleanNode,
                        int2name: dict[int, str] | None = None) -> str:
    """Transforms the prime implicants LUT of a Boolean Node from CANA to algebraic format.

    Parameters
    ----------
    node : BooleanNode
        CANA Boolean node. See: https://github.com/rionbr/CANA
    int2name : dict[int, str], optional
        Dictionary with the node ids as keys and node name as values, by default None

    Returns
    -------
    str
        Node rule in algebraic format.
        Ex.: A* = A|B&C
    """
    if int2name == None:
        int2name = {i: "x{}".format(i) for i in node.inputs}

    if node.constant:
        return "{name}* = {state}".format(name=int2name[node.id], state=node.state)

    node._check_compute_canalization_variables(prime_implicants=True)

    if node.bias() < 0.5:
        alg_rule = "{name}* = ".format(name=int2name[node.id])
        prime_rules = node._prime_implicants['1']
    else:
        alg_rule = "{name}* = !(".format(name=int2name[node.id])
        prime_rules = node._prime_implicants['0']

    for rule in prime_rules:
        for k, out in enumerate(rule):
            if out == '1':
                alg_rule += "{name}&".format(name=int2name[node.inputs[k]])
            elif out == '0':
                alg_rule += "!{name}&".format(name=int2name[node.inputs[k]])
        alg_rule = alg_rule[:-1]+"|"

    if node.bias() < 0.5:
        alg_rule = alg_rule[:-1]
    else:
        alg_rule = alg_rule[:-1]+")"

    return alg_rule


def network_rules_from_cana(BN: cana.BooleanNetwork) -> str:
    """Transforms the prime implicants LUT of a Boolean Network from CANA to algebraic format.

    ----------
    BN : BooleanNetwork
        CANA Boolean network. See: https://github.com/rionbr/CANA

    Returns
    -------
    str
        Network rules in algebraic format.
        Ex.: A* = A|B&C\nB* = C\nC* = A|B
    """

    alg_rule = ""
    int2name = {v: parser.name_adjustment(k) for k, v in BN.name2int.items()}
    for node in BN.nodes:
        alg_rule += node_rule_from_cana(node=node, int2name=int2name)
        alg_rule += "\n"

    return alg_rule[:-1]
