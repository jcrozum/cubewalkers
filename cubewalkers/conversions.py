"""This module contains utilities for converting between Boolean network formats."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import cupy as cp  # type: ignore

from cubewalkers import parser

if TYPE_CHECKING:
    from cana.boolean_network import BooleanNetwork  # type: ignore
    from cana.boolean_node import BooleanNode  # type: ignore


def cana2cupyLUT(
    net: BooleanNetwork,
) -> tuple[cp.NDArray, cp.NDArray]:
    """
    Extract lookup tables and input lists from a CANA network into a CuPy-compatible form.

    For more information about CANA, see: https://github.com/rionbr/CANA.

    Parameters
    ----------
    net : cana.BooleanNetwork
        CANA network to import.

    Returns
    -------
    tuple[NDArray, NDArray]
        Returns a merged lookup table that contains the output column of each rule's
        lookup table (padded by False values). Also returns and inputs table, which
        contains the inputs for each node (padded by -1).
    """
    network_nodes: list[BooleanNode] = net.nodes  # type: ignore
    outs = [u.outputs for u in network_nodes]
    outmax = len(max(outs, key=lambda x: len(x)))
    outs_bool = [
        list(map(lambda x: x == "1", out)) + [False] * (outmax - len(out))
        for out in outs
    ]  # convert to booland pad for cupy

    inps = [u.inputs for u in network_nodes]
    inpmax = len(max(inps, key=lambda x: len(x)))
    inps_pad = [
        inp + [-1] * (inpmax - len(inp)) for inp in inps
    ]  # convert to booland pad for cupy

    out_columns: cp.NDArray = cp.array(outs_bool, dtype=cp.bool_)  # type: ignore
    in_columns: cp.NDArray = cp.array(inps_pad, dtype=cp.int32)  # type: ignore
    return out_columns, in_columns


def cana2cupy_probabilisticLUT(
    net: BooleanNetwork, prob: float = 0.0
) -> tuple[cp.NDArray, cp.NDArray]:
    """
    Extract probabilistic lookup tables and input lists from a CANA network into a CuPy-compatible form.

    For more information about CANA, see: https://github.com/rionbr/CANA.

    Parameters
    ----------
    net : cana.BooleanNetwork
        CANA network to import.

    Returns
    -------
    tuple[NDArray, NDArray]
        Returns a merged lookup table that contains the float output column of each rule's
        lookup table, where "1" is reduced by prob, and "0" is increasd by prob.
        i.e. if the prob = 0.01, then the LUT outputs are 0.01 for False and 0.99 for True.
        The LUT is padded by False values (0 + prob) so every LUT is of the same size.
        Also returns an inputs table, which contains the inputs for each node (padded by -1).
    """
    network_nodes: list[BooleanNode] = net.nodes  # type: ignore
    outs = [u.outputs for u in network_nodes]
    outmax = len(max(outs, key=lambda x: len(x)))

    outs_float = [
        [1 - prob if x == "1" else prob for x in out]
        + [False] * (outmax - len(out))  # padded for cupy
        for out in outs
    ]
    inps = [u.inputs for u in network_nodes]
    inpmax = len(max(inps, key=lambda x: len(x)))
    inps_pad = [inp + [-1] * (inpmax - len(inp)) for inp in inps]
    out_columns: cp.NDArray = cp.array(outs_float, dtype=cp.float32)  # type: ignore
    in_columns: cp.NDArray = cp.array(inps_pad, dtype=cp.int32)  # type: ignore
    return out_columns, in_columns


def node_rule_from_cana(
    node: BooleanNode, int2name: dict[int, str] | None = None
) -> str:
    """
    Transforms the prime implicants LUT of a Boolean Node from CANA to algebraic format.

    For more information about CANA, see: https://github.com/rionbr/CANA.

    Parameters
    ----------
    node : BooleanNode
        CANA Boolean node.
    int2name : dict[int, str], optional
        Dictionary with the node ids as keys and node name as values, by default None

    Returns
    -------
    str
        Node rule in algebraic format.
    """
    if int2name is None:
        int2name = {i: "x{}".format(i) for i in node.inputs}

    if node.constant:
        return f"{int2name[node.id]}* = {node.state}"  # type: ignore

    node._check_compute_canalization_variables(prime_implicants=True)  # type: ignore

    if node.bias() <= 0.5:
        alg_rule = "{name}* = ".format(name=int2name[node.id])
        prime_rules: Iterable[Iterable[str]] = node._prime_implicants["1"]  # type: ignore
    else:
        alg_rule = "{name}* = !(".format(name=int2name[node.id])
        prime_rules: Iterable[Iterable[str]] = node._prime_implicants["0"]  # type: ignore

    for rule in prime_rules:  # type: ignore
        assert isinstance(rule, str)
        for k, out in enumerate(rule):
            if out == "1":
                alg_rule += "{name}&".format(name=int2name[node.inputs[k]])
            elif out == "0":
                alg_rule += "!{name}&".format(name=int2name[node.inputs[k]])
        alg_rule = alg_rule[:-1] + "|"

    if node.bias() <= 0.5:
        alg_rule = alg_rule[:-1]
    else:
        alg_rule = alg_rule[:-1] + ")"

    return alg_rule


def network_rules_from_cana(BN: BooleanNetwork) -> str:
    """
    Transforms the prime implicants LUT of a Boolean Network from CANA to algebraic format.

    For more information about CANA, see: https://github.com/rionbr/CANA.

    Parameters
    ----------
    BN : BooleanNetwork
        CANA Boolean network. See: https://github.com/rionbr/CANA

    Returns
    -------
    str
        Network rules in algebraic format.
    """

    alg_rule = ""
    name2int: dict[str, int] = BN.name2int  # type: ignore
    int2name = {v: parser.name_adjustment(k) for k, v in name2int.items()}
    nodes: list[BooleanNode] = BN.nodes  # type: ignore
    for node in nodes:
        alg_rule += node_rule_from_cana(node=node, int2name=int2name)
        alg_rule += "\n"

    return alg_rule[:-1]


def cpp2bnet(cpp_rules: str) -> str:
    """
    Converts the rules in C++ format to bnet format.

    Parameters
    ----------
    cpp_rules : str
        rules in the C++ form.

    Returns
    -------
    str
        rules in the bnet form.
    """
    bnet_rules = ""
    for line in cpp_rules.split("\n"):
        if not line.startswith("#"):
            line = line.replace("&&", "&")
            line = line.replace("||", "|")
        bnet_rules += line
        bnet_rules += "\n"
    return bnet_rules
