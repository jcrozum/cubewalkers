"""This module contains utilities for generating initial conditions for Boolean networks."""

from __future__ import annotations

from io import StringIO
from typing import Iterable

import cupy as cp  # type: ignore


def initial_walker_states(
    initial_biases: str,
    vardict: dict[str, int],
    n_walkers: int,
    comment_char: str = "#",
) -> cp.NDArray:
    """
    Generates initial conditions for an ensemble of walkers.

    Parameters
    ----------
    initial_biases : str
        Each line should be of the form

        * NodeName,Bias

        where NodeName is the name of the node, and Bias is the probability that the node
        will be initialized to 1 (instead of 0). Nodes whose names are not listed are given
        a bias of 0.5 by default.
    vardict : dict[str,int]
        A dictionary whose keys are the node names of the Boolean network and whose values
        are the corresponding indicies describing the ordering of the variables.
    n_walkers : int
        The number of walkers to initialize.
    comment_char : str, optional
        Empty lines and lines beginning with this character are ignored, by default '#'.

    Returns
    -------
    cp.NDarray
        n_variables x n_walkers array of initial states.
    """
    # get just the relevant lines from the input
    lines: Iterable[str] = list(StringIO(initial_biases))
    lines = [line.strip().lstrip() for line in lines]
    lines = filter(lambda x: not x.startswith(comment_char), lines)
    lines = filter(lambda x: not x.startswith("\n"), lines)
    lines = filter(lambda x: not x == "", lines)

    bias_dict = {
        vardict[x.split(",")[0].strip().lstrip()]: float(x.split(",")[1]) for x in lines
    }

    initial_states: cp.NDArray = cp.random.choice(  # type: ignore
        [cp.bool_(0), cp.bool_(1)], (len(vardict), n_walkers)
    )

    for w_ind in range(n_walkers):
        for i, v in bias_dict.items():
            initial_states[i, w_ind] = cp.bool_(
                (cp.random.random() <= v).get()  # type: ignore # NDArray of floats mapped to bool_
            )

    return initial_states
