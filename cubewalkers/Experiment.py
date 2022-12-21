from io import StringIO
from typing import Iterable
from cubewalkers import parser
import re


class Experiment:
    """Stores information from user-specified experimental inputs.
    """

    def __init__(self, override_string: str, comment_char: str = '#') -> None:
        """_summary_

        Parameters
        ----------
        override_string : str
            Experimental conditions to incorporate. Each line should be of the form

            NodeName,StartTime,EndTime,RuleToSubstitute
            
            - or -
            
            ParentName-->NodeName,StartTime,EndTime,RuleToSubstitute

            If NodeName ends in a '*', then update is not forced (i.e., the update
            rule becomes fixed to RuleToSubstitute). Otherwise, update is forced,
            and the node NodeName takes the value RuleToSubstitute at each time 
            step. Note that this does not make any difference in synchronous update.
            The control can be made permanent by specifying an EndTime of inf.
            If NodeName is preceded by 'ParentName-->', then RuleToSubstitute replaces
            all occurences of ParentName in the update rule for NodeName, and the
            update is never forced. If conflicting controls are specified, ealier
            rows take precedence.
        comment_char : str, optional
            Empty lines and lines beginning with this character are ignored, by default '#'.
        """
        override_list = sorted(StringIO(override_string))
        self.overrides = {}
        self.edge_subs = {}
        self.force_update_time_strings = {}

        for s in override_list:
            s = s.lstrip()
            if len(s.strip()) == 0 or s[0] == comment_char:
                continue

            varname, ti, tf, rule_override = [x.strip() for x in s.split(',')]
            rule_override = parser.clean_rules(rule_override).strip()
            if tf == 'inf': # no infinite builtin in cpp/cuda, so we use this hack
                tf = 't__reserved' # perpetually controlling for "just one more timestep"
            
            edgetic = '-->' in varname
            if edgetic:
                parent,varname = [x.strip().lstrip() for x in varname.split('-->')]
            else:
                parent = None
            
            force_update = not edgetic # no forced update in edgetic control
            if varname[-1] == '*':
                varname = varname[:-1]
                force_update = False

            if edgetic:
                edge_sub_string = (
                    f'(t__reserved < {ti} || t__reserved > {tf}) && '
                    f'{parent}'
                    f'|| (t__reserved >= {ti} && t__reserved <= {tf}) && '
                    f'({rule_override}) '
                )
                if varname not in self.edge_subs.keys():
                    self.edge_subs[varname] = [(parent,edge_sub_string)]
                else:
                    self.edge_subs[varname].append((parent,edge_sub_string))
            else:
                if varname not in self.overrides.keys():
                    self.overrides[varname] = '( original_rule__reserved ) '
                    self.force_update_time_strings[varname] = ''
                else:
                    self.force_update_time_strings[varname] += ' || '

                self.overrides[varname] = (
                    f'(t__reserved < {ti} || t__reserved > {tf}) && '
                    f'{self.overrides[varname]}'
                    f'|| (t__reserved >= {ti} && t__reserved <= {tf}) && '
                    f'({rule_override}) '
                )
                if force_update:
                    self.force_update_time_strings[varname] += (
                        f'(t__reserved >= {ti} && t__reserved <= {tf})'
                    )
                else:
                    self.force_update_time_strings[varname] += 'false'
            
                

    def new_rule(self, old_rule: str) -> str:
        """Modifies an input rule to incorporate the time-dependent experimental conditions
        stored internally.

        Parameters
        ----------
        old_rule : str
            Rule to modify.

        Returns
        -------
        str
            Modified rule, with time dependence.
        """
        varname, old_function = [x.strip() for x in old_rule.split(',')]
        new_function = old_function
        
        if varname in self.overrides.keys():
            new_function = re.sub("original_rule__reserved",
                                  old_function, self.overrides[varname])
            
        if varname in self.edge_subs.keys():
            for parent,parent_sub in self.edge_subs[varname]:
                new_function = re.sub(fr'\b{parent}\b',
                            parent_sub,
                            new_function)
        
        return varname + ',\t' + new_function + '\n'

    def time_clamp_string(self, varnames: Iterable[str]) -> str:
        """Generates and returns a string that determines when the update mask should
        be ignored because of the experimental conditions.

        Parameters
        ----------
        varnames : Iterable[str]
            Names of the variables in the system.

        Returns
        -------
        str
            String that is interperable as a C++ Boolean expression. When this evaluates
            to true, the update mask is ignored and an update is performed.
        """
        time_clamp_string_parts = ['']
        for i, v in enumerate(varnames):
            if v in self.force_update_time_strings:
                force_string = self.force_update_time_strings[v]
                time_clamp_string_parts.append(
                    f'((n__reserved=={i}) && ({force_string}))')
        return ' || '.join(time_clamp_string_parts)
