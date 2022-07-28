from __future__ import annotations
import cupy as cp
from io import StringIO
import re

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Experiment import Experiment


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

    cpp_body = '''extern "C" __global__
void {}(const bool* A__reserved_input,
        const float* A__reserved_mask,
        bool* A__reserved_output,
        int t__reserved, int N__reserved, int W__reserved) {{
    int w__reserved = blockDim.x * blockIdx.x + threadIdx.x;
    int n__reserved = blockDim.y * blockIdx.y + threadIdx.y;
    int a__reserved = w__reserved + n__reserved*W__reserved;
    if(n__reserved < N__reserved && w__reserved < W__reserved){{
        if(A__reserved_mask[a__reserved]>0{}){{'''.format(
        kernel_name, time_clamp_string)

    for i, v in enumerate(varnames):
        s = re.sub(r'\b{}\b'.format(
            v), 'A__reserved_input[{}*W__reserved+w__reserved]'.format(i), s)

    for ln, line in enumerate(StringIO(s)):
        update_function_string = line.split(',')[1].strip()
        cpp_body += ('\n            if (n__reserved=={}){{A__reserved_output[a__reserved]='
                     '{};}}'.format(
                         ln, update_function_string)
                     )

    cpp_body += '\n} else{A__reserved_output[a__reserved]=A__reserved_input[a__reserved];}}}'

    return cp.RawKernel(cpp_body, kernel_name), varnames, cpp_body
