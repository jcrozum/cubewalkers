"""This module contains methods for constructing CUDA kernels from Boolean networks."""

from __future__ import annotations

import re
from io import StringIO
from typing import TYPE_CHECKING, Iterable

import cupy as cp  # type: ignore

if TYPE_CHECKING:
    from cubewalkers.custom_typing import RawKernelType
    from cubewalkers.experiment import Experiment


def clean_rules(rules: str, comment_char: str = "#") -> str:
    """
    Initial parsing of rules to standardize format.

    Parameters
    ----------
    rules : str
        Rule string to parse.
    comment_char : str, optional
        Empty lines and lines beginning with this character are ignored, by default `#`.

    Returns
    -------
    str
        A reformatted version of the input rules.
    """

    # if we're already using cpp logical operators, switch to single for now
    s = re.sub(r"\|\|", "|", rules)
    s = re.sub(r"\&\&", "&", s)
    # make sure we're in proper bnet format
    s = re.sub(r"\s*\*\s*=\s*", ",\t", s)  # replace "*=" with ",\t"
    s = re.sub(r"\s+not\s+", " !", s, flags=re.IGNORECASE)  # not -> !
    # not -> ! (with parens)
    s = re.sub(r"\(\s*not\s+", "(!", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*~\s*", " !", s, flags=re.IGNORECASE)  # ~ -> !
    s = re.sub(r"\s+and\s+", " & ", s, flags=re.IGNORECASE)  # and -> &
    s = re.sub(r"\s+or\s+", " | ", s, flags=re.IGNORECASE)  # or -> |
    # False -> 0 (ignore case)
    s = re.sub("False", "0", s, flags=re.IGNORECASE)
    s = re.sub("True", "1", s, flags=re.IGNORECASE)  # True -> 1 (ignore case)

    # now switch to cpp logical operators
    s = re.sub(r"\s*\|\s*", " || ", s)
    s = re.sub(r"\s*\&\s*", " && ", s)

    # PBN support
    s = re.sub(
        "[<][<][=]",
        " < A__reserved_mask[a__reserved] && A__reserved_mask[a__reserved] <= ",
        s,
    )

    # filter comments, blank lines
    lines: Iterable[str] = list(StringIO(s))
    lines = [line.lstrip() for line in lines]
    lines = filter(lambda x: not x.startswith(comment_char), lines)
    lines = filter(lambda x: not x.startswith("\n"), lines)
    lines = filter(lambda x: not x == "", lines)

    # reassemble
    return "".join(lines)


def adjust_rules_for_experiment(rules: str, experiment: Experiment | None) -> str:
    """
    Helper function that adjusts rules to incorporate the experimental conditions
    specified in the experiment string.

    Parameters
    ----------
    rules : str
        Rules to adjust. Assumes these have been cleaned.
    experiment : Experiment | None
        Experiment object to use for rule adjustment; if None, do nothing.

    Returns
    -------
    str
        A new rules string that includes time-dependent modifications according to the
        experiment string.
    """
    if experiment is None:
        return rules

    lines: list[str] = []
    for line in StringIO(rules):
        lines.append(experiment.new_rule(line))

    return "".join(lines)


def bnet2rawkernel(
    rules: str,
    kernel_name: str,
    experiment: Experiment | None = None,
    comment_char: str = "#",
    skip_clean: bool = False,
) -> tuple[RawKernelType, list[str], str]:
    """
    Generates a CuPy RawKernel that encodes update rules and experimental conditions.

    Parameters
    ----------
    rules : str
        Rules to input. If skip_clean is True (not default), then these are assumed to have
        been cleaned.
    kernel_name : str
        A name for the kernel
    experiment : Experiment | None, optional
        An Experiment object specifying experimental conditions, by default None, in which
        case no experimental conditions are incorporated into the rules.
    comment_char : str, optional
        In rules, empty lines and lines beginning with this character are ignored, by default
        `#`.
    skip_clean : bool, optional
        Whether to skip the step of cleaning the rules, by default False.

    Returns
    -------
    RawKernelType
        A CuPy RawKernel that is called to compute time evolution.
    list[str]
        A list of strings that encode the variable names
    str
        A string containing the CUDA-C++ body of the kernel
    """
    if not skip_clean:
        s = clean_rules(rules, comment_char=comment_char)
    else:
        s = rules

    # construct actual cpp code from reformatted rules
    varnames = [line.split(",")[0] for line in StringIO(s)]
    # vardict = {k:i for i,k in enumerate(varnames)}

    # Now, modify rules to incorporate experiment procedures
    s = adjust_rules_for_experiment(s, experiment)

    # account for force updates due to control
    time_clamp_string = ""
    if experiment is not None:
        time_clamp_string = experiment.time_clamp_string(varnames)

    cpp_body = (
        f'extern "C" __global__\n'
        f"void {kernel_name}(const bool* A__reserved_input,\n"
        f"        const float* A__reserved_mask,\n"
        f"        bool* A__reserved_output,\n"
        f"        int t__reserved, int N__reserved, int W__reserved) {{\n"
        f"    int w__reserved = blockDim.x * blockIdx.x + threadIdx.x;\n"
        f"    int n__reserved = blockDim.y * blockIdx.y + threadIdx.y;\n"
        f"    int a__reserved = w__reserved + n__reserved*W__reserved;\n"
        f"    if(n__reserved < N__reserved && w__reserved < W__reserved){{\n"
        f"        if(A__reserved_mask[a__reserved]>0{time_clamp_string}){{"
    )
    for i, v in enumerate(varnames):
        s = re.sub(rf"\b{v}\b", f"A__reserved_input[{i}*W__reserved+w__reserved]", s)

    for ln, line in enumerate(StringIO(s)):
        update_function_string = line.split(",")[1].strip()
        cpp_body += (
            f"\n            if (n__reserved=={ln})"
            f"{{A__reserved_output[a__reserved]={update_function_string};}}"
        )

    cpp_body += (
        "\n} else{A__reserved_output[a__reserved]=A__reserved_input[a__reserved];}}}"
    )

    kernel: RawKernelType = cp.RawKernel(cpp_body, kernel_name)  # type: ignore

    return kernel, varnames, cpp_body  # type: ignore


def regulators2lutkernel(
    node_regulators: Iterable[Iterable[int]], kernel_name: str
) -> tuple[RawKernelType, str]:
    """
    Constructs a CuPy RawKernel that simulates a network using lookup tables.

    Parameters
    ----------
    node_regulators : Iterable[Iterable[int]]
        Iterable i should contain the indicies of the nodes that regulate node i,
        optionally padded by negative values.
    kernel_name : str
        Name of the kernel to be generated.

    Returns
    -------
    RawKernelType
        A CuPy RawKernel that is called to compute time evolution.
    str
        A string containing the CUDA-C++ body of the kernel
    """
    cpp_body = (
        f'extern "C" __global__\n'
        f"void {kernel_name}(const bool* A__reserved_input,\n"
        f"        const float* A__reserved_mask,\n"
        f"        bool* A__reserved_output,\n"
        f"        const bool* A__reserved_LUT,\n"
        f"        int t__reserved, int N__reserved, int W__reserved, int L__reserved) {{\n"
        f"    int w__reserved = blockDim.x * blockIdx.x + threadIdx.x;\n"
        f"    int n__reserved = blockDim.y * blockIdx.y + threadIdx.y;\n"
        f"    int a__reserved = w__reserved + n__reserved*W__reserved;\n"
        f"    if(n__reserved < N__reserved && w__reserved < W__reserved){{\n"
        f"        if(A__reserved_mask[a__reserved]>0){{\n"
    )

    for ln, input in enumerate(node_regulators):
        lookup_index = "0"
        for ind in input:
            if ind < 0:
                break
            lookup_index = (
                f"({lookup_index}<<1)"
                f"+A__reserved_input[{ind}*W__reserved+w__reserved]"
            )

        cpp_body += (
            f"if (n__reserved=={ln}){{"
            f"A__reserved_output[a__reserved]="
            f"A__reserved_LUT[({lookup_index})+L__reserved*n__reserved];}}\n"
        )
    cpp_body += (
        "\n} else{A__reserved_output[a__reserved]=A__reserved_input[a__reserved];}}}"
    )

    # print(cpp_body)
    kernel: RawKernelType = cp.RawKernel(cpp_body, kernel_name)  # type: ignore
    return kernel, cpp_body  # type: ignore

def regulators2probabilisticlutkernel(
    node_regulators: Iterable[Iterable[int]], kernel_name: str, 
) -> tuple[RawKernelType, str]:
    """
    Constructs a CuPy RawKernel that simulates a network using lookup tables.
    Same as regulators2lutkernel, but with float LUTs for a probabilistic network.

    Parameters
    ----------
    node_regulators : Iterable[Iterable[int]]
        Iterable i should contain the indicies of the nodes that regulate node i,
        optionally padded by negative values.
    kernel_name : str
        Name of the kernel to be generated.

    Returns
    -------
    RawKernelType
        A CuPy RawKernel that is called to compute time evolution.
    str
        A string containing the CUDA-C++ body of the kernel
    """

    cpp_body = (
        f'extern "C" __global__\n'
        f"void {kernel_name}(const bool* A__reserved_input,\n"
        f"        const float* A__reserved_mask,\n"
        f"        bool* A__reserved_output,\n"
        f"        const float* A__reserved_LUT,\n" # CHANGED BOOL TO FLOAT
        f"        int t__reserved, int N__reserved, int W__reserved, int L__reserved) {{\n"
        f"    int w__reserved = blockDim.x * blockIdx.x + threadIdx.x;\n"
        f"    int n__reserved = blockDim.y * blockIdx.y + threadIdx.y;\n"
        f"    int a__reserved = w__reserved + n__reserved*W__reserved;\n"
        f"    if(n__reserved < N__reserved && w__reserved < W__reserved){{\n"
        f"        if(A__reserved_mask[a__reserved]>0){{\n" 
    )

    for ln, regulator_input in enumerate(node_regulators):
        lookup_index = "0"
        for ind in regulator_input:
            if ind < 0:
                break
            lookup_index = (
                f"({lookup_index}<<1)"
                f"+A__reserved_input[{ind}*W__reserved+w__reserved]"
            )

        cpp_body += (
            f"if (n__reserved=={ln}){{"
            f"A__reserved_output[a__reserved]="
            f"(A__reserved_LUT[({lookup_index})+L__reserved*n__reserved]>= A__reserved_mask[a__reserved]);}}\n"
        )
    cpp_body += (
        "\n} else{A__reserved_output[a__reserved]=A__reserved_input[a__reserved];}}}"
    )

    # print(cpp_body)
    kernel: RawKernelType = cp.RawKernel(cpp_body, kernel_name)  # type: ignore
    return kernel, cpp_body  # type: ignore

def name_adjustment(name: str) -> str:
    """
    Adjust the node name to fit proper formatting.

    Parameters
    ----------
    name : str
        Original name of the node.

    Returns
    -------
    str
        Adjusted name of the node.
    """

    not_allowed = ["-", ")", "(", "/"]

    for expression in not_allowed:
        name = name.replace(expression, "_")

    name = name.replace("\\", "")
    name = name.replace("+", "_plus_")

    if name[0].isdigit():
        name = "number_" + name

    return name
