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
            
            If NodeName ends in a '*', then update is not forced (i.e., the update
            rule becomes fixed to RuleToSubstitute). Otherwise, update is forced,
            and the node NodeName takes the value RuleToSubstitute at each time 
            step. Note that this does not make any difference in synchronous update.
        comment_char : str, optional
            Empty lines and lines beginning with this character are ignored, by default '#'.
        """
        override_list = sorted(StringIO(override_string))
        self.overrides = {}
        self.force_update_time_strings = {}
        
        for s in override_list:
            s = s.lstrip()
            if len(s.strip()) == 0 or s[0] == comment_char:
                continue

            varname, ti, tf, rule_override = [x.strip() for x in s.split(',')]
            rule_override = parser.clean_rules(rule_override).strip()
            
            force_update = True
            if varname[-1] == '*':
                varname = varname[:-1]
                force_update = False

            if varname not in self.overrides.keys():
                self.overrides[varname] = '( original_rule__reserved ) '
                self.force_update_time_strings[varname] = ''
            else:
                self.force_update_time_strings[varname] += ' || '
                
            if tf == 'inf':
                self.overrides[varname] = (
                    f'(t__reserved < {ti}) && '
                    f'{self.overrides[varname]}'
                    f'|| (t__reserved >= {ti}) && {rule_override}'
                )
                
                if force_update:
                    self.force_update_time_strings[varname] += (
                        f'(t__reserved >= {ti})'
                    )
                else:
                    self.force_update_time_strings[varname] += 'false'
            else:
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
        if varname not in self.overrides.keys():
            return old_rule
        else:
            new_function = re.sub("original_rule__reserved",
                                  old_function, self.overrides[varname])
            new_rule = varname + ',\t' + new_function + '\n'
            return new_rule

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