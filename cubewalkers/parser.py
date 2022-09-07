from __future__ import annotations
import cupy as cp
from io import StringIO
import re

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Experiment import Experiment
    from cana.boolean_network import BooleanNetwork, BooleanNode


def clean_rules(rules: str, comment_char: str = '#') -> str:
    """Initial parsing of rules to standardize format.

    Parameters
    ----------
    rules : str
        Rule string to parse.
    comment_char : str, optional
        Empty lines and lines beginning with this character are ignored, by default '#'.

    Returns
    -------
    str
        A reformatted version of the input rules.
    """

    # make sure we're in proper bnet format
    s = re.sub("\s*\*\s*=\s*", ",\t", rules)  # replace "*=" with ",\t"
    s = re.sub("\s+not\s+", " !", s, flags=re.IGNORECASE)  # not -> !
    # not -> ! (with parens)
    s = re.sub("\(\s*not\s+", "(!", s, flags=re.IGNORECASE)
    s = re.sub("\s*~\s*", " !", s, flags=re.IGNORECASE)  # ~ -> !
    s = re.sub("\s+and\s+", " & ", s, flags=re.IGNORECASE)  # and -> &
    s = re.sub("\s+or\s+", " | ", s, flags=re.IGNORECASE)  # or -> |
    # False -> 0 (ignore case)
    s = re.sub("False", "0", s, flags=re.IGNORECASE)
    s = re.sub("True", "1", s, flags=re.IGNORECASE)  # True -> 1 (ignore case)

    # now switch to cpp logical operators
    s = re.sub("\s*\|\s*", " || ", s)
    s = re.sub("\s*\&\s*", " && ", s)

    # PBN support
    s = re.sub("[<][<][=]",
               " < A__reserved_mask[a__reserved] && A__reserved_mask[a__reserved] <= ",
               s)

    # filter comments, blank lines
    lines = list(StringIO(s))
    lines = [l.lstrip() for l in lines]
    lines = filter(lambda x: not x.startswith(comment_char), lines)
    lines = filter(lambda x: not x.startswith('\n'), lines)
    lines = filter(lambda x: not x == '', lines)

    # reassemble
    return "".join(lines)


def adjust_rules_for_experiment(rules: str, experiment: str) -> str:
    """Helper function that adjusts rules to incorporate the experimental conditions
    specified in the experiment string.

    Parameters
    ----------
    rules : str
        Rules to adjust. Assumes these have been cleaned.
    experiment : str
        Experimental conditions to incorporate. Each line should be of the form
        NodeName,StartTime,EndTime,RuleToSubstitute

    Returns
    -------
    str
        A new rules string that includes time-dependent modifications according to the
        experiment string.
    """
    if experiment is None:
        return rules

    lines = []
    for line in StringIO(rules):
        lines.append(experiment.new_rule(line))

    return "".join(lines)


def bnet2rawkernel(rules: str,
                   kernel_name: str,
                   experiment: Experiment | None = None,
                   comment_char: str = '#',
                   skip_clean: bool = False) -> cp.RawKernel:
    """Generates a CuPy RawKernel that encodes update rules and experimental conditions.

    Parameters
    ----------
    rules : str
        Rules to input. If skip_clean is True (not default), then these are assumed to have
        been cleaned.
    kernel_name : str
        A name for the kernel
    experiment : Experiment | None, optional
        A string specifying experimental conditions, by default None, in which case no 
        experimental conditions are incorporated into the rules.
    comment_char : str, optional
        In rules, empty lines and lines beginning with this character are ignored, by default
        '#'.
    skip_clean : bool, optional
        Whether to skip the step of cleaning the rules, by default False.

    Returns
    -------
    cp.RawKernel
        A CuPy RawKernel that accepts arguments in the following fashion:
        kernel(blocks_per_grid, threads_per_block, (
            input_array_to_update, 
            update_scheme_mask, 
            output_array_after_update, 
            current_time_step, 
            number_of_nodes, 
            number_of_walkers))
        and modifies the output_array_after_update in-place.
    """
    if not skip_clean:
        s = clean_rules(rules, comment_char=comment_char)
    else:
        s = rules

    # construct actual cpp code from reformatted rules
    varnames = tuple(line.split(',')[0] for line in StringIO(s))
    #vardict = {k:i for i,k in enumerate(varnames)}

    # Now, modify rules to incorporate experiment procedures
    s = adjust_rules_for_experiment(s, experiment)

    # account for force updates due to control
    time_clamp_string_parts = ['']
    for i, v in enumerate(varnames):
        if experiment is not None:
            if v in experiment.force_update_time_strings:
                force_string = experiment.force_update_time_strings[v]
                time_clamp_string_parts.append(
                    '(n__reserved=={} && ({}))'.format(i, force_string))
    time_clamp_string = ' || '.join(time_clamp_string_parts)

    cpp_body = (
        f'extern "C" __global__\n'
        f'void {kernel_name}(const bool* A__reserved_input,\n'
        f'        const float* A__reserved_mask,\n'
        f'        bool* A__reserved_output,\n'
        f'        int t__reserved, int N__reserved, int W__reserved) {{\n'
        f'    int w__reserved = blockDim.x * blockIdx.x + threadIdx.x;\n'
        f'    int n__reserved = blockDim.y * blockIdx.y + threadIdx.y;\n'
        f'    int a__reserved = w__reserved + n__reserved*W__reserved;\n'
        f'    if(n__reserved < N__reserved && w__reserved < W__reserved){{\n'
        f'        if(A__reserved_mask[a__reserved]>0{time_clamp_string}){{'
    )
    for i, v in enumerate(varnames):
        s = re.sub(fr'\b{v}\b',
                   f'A__reserved_input[{i}*W__reserved+w__reserved]',
                   s)

    for ln, line in enumerate(StringIO(s)):
        update_function_string = line.split(',')[1].strip()
        cpp_body += (
            f'\n            if (n__reserved=={ln})'
            f'{{A__reserved_output[a__reserved]={update_function_string};}}'
        )

    cpp_body += '\n} else{A__reserved_output[a__reserved]=A__reserved_input[a__reserved];}}}'

    return cp.RawKernel(cpp_body, kernel_name), varnames, cpp_body

def network_rules_from_cana(BN: BooleanNetwork) -> str:
    """Transforms the prime implicants LUT of a Boolean Network from CANA to algebraic format.

    Parameters: https://github.com/rionbr/CANA
    ----------
    BN : BooleanNetwork
        CANA Boolean network

    Returns
    -------
    str
        Network rules in algebraic format.
        Ex.: A* = A|B&C\nB* = C\nC* = A|B
    """
    
    alg_rule = ""
    int2name = {v: name_adjustment(k) for k, v in BN.name2int.items()}
    for node in BN.nodes:
        alg_rule += node_rule_from_cana(node=node, int2name=int2name)
        alg_rule += "\n"
    
    return alg_rule[:-1]

def node_rule_from_cana(node: BooleanNode, int2name: dict[int, str] | None = None) -> str:
    """Transforms the prime implicants LUT of a Boolean Node from CANA to algebraic format.

    Parameters
    ----------
    node : BooleanNode
        CANA Boolean node
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

def name_adjustment(name: str) -> str:
    """Adjust the node name to fit proper formatting.

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
        name = 'number_' + name
    
    return name