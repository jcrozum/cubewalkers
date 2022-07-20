from io import StringIO
from cubewalkers import parser
import re


class Experiment:
    def __init__(self, override_string: str, comment_char: str = '#') -> None:
        override_list = sorted(StringIO(override_string))
        self.overrides = {}
        self.force_update_time_strings = {}
        for s in override_list:
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
                self.overrides[varname] = ('(t__reserved < {}) && '.format(ti) +
                                           self.overrides[varname] +
                                           '|| (t__reserved >= {}) && {}'.format(ti, rule_override))
                self.force_update_time_strings[varname] += '(t__reserved >= {})'.format(ti)
            else:
                self.overrides[varname] = ('(t__reserved < {} || t__reserved > {}) && '.format(ti, tf) +
                                           self.overrides[varname] +
                                           '|| (t__reserved >= {} && t__reserved <= {}) && ({}) '.format(ti, tf, rule_override))
                self.force_update_time_strings[varname] += '(t__reserved >= {} && t__reserved <= {})'.format(ti, tf)
    def new_rule(self, old_rule: str) -> str:
        varname, old_function = [x.strip() for x in old_rule.split(',')]
        if varname not in self.overrides.keys():
            return old_rule
        else:
            new_function = re.sub("original_rule__reserved",
                                  old_function, self.overrides[varname])
            new_rule = varname + ',\t' + new_function + '\n'
            # print(f'{old_rule=}\n{new_rule=}')
            return new_rule
