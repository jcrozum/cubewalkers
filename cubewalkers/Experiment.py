from io import StringIO
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
                
                self.force_update_time_strings[varname] += (
                    '(t__reserved >= {})'.format(ti)
                )
            else:
                self.overrides[varname] = (
                    f'(t__reserved < {ti} || t__reserved > {tf}) && '
                    f'{self.overrides[varname]}'
                    f'|| (t__reserved >= {ti} && t__reserved <= {tf}) && '
                    f'({rule_override}) '
                )

                self.force_update_time_strings[varname] += (
                    f'(t__reserved >= {ti} && t__reserved <= {tf})'
                )

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
