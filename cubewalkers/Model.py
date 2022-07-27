from __future__ import annotations
from cubewalkers import simulation, parser

# for generating model names when the user doesn't want to specify them
import string
import random

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Experiment import Experiment


class Model():
    _automatic_model_name_length = 16
    _automatic_model_names_taken = set()

    def __init__(self,
                 rules: str,
                 model_name: str | None = None,
                 experiment: Experiment | None = None,
                 comment_char: str = '#',
                 n_time_steps: int = 1,
                 n_walkers: int = 1) -> None:
        if model_name is None:
            g = None
            while g is None or g in Model._automatic_model_names_taken:
                g = ''.join(random.choices(string.ascii_letters,
                            k=Model._automatic_model_name_length))
            self.name = g
            Model._automatic_model_names_taken.add(self.name)
        else:
            self.name = model_name
        self.rules = parser.clean_rules(rules, comment_char=comment_char)
        self.kernel, self.varnames, self.code = parser.bnet2rawkernel(
            self.rules, self.name, experiment=experiment, skip_clean=True)
        self.vardict = {k: i for i, k in enumerate(self.varnames)}
        self.n_time_steps = n_time_steps
        self.n_walkers = n_walkers
        self.n_variables = len(self.varnames)

    def simulate_random_ensemble(self, n_time_steps: int = None,
                                 n_walkers: int = None,
                                 averages_only: bool = False,
                                 maskfunction: callable | None = None,
                                 threads_per_block: tuple[int, int] = (32, 32)) -> None:
        if n_time_steps is not None:
            self.n_time_steps = n_time_steps
        if n_walkers is not None:
            self.n_walkers = n_walkers
        
        self.averages_only = averages_only
        
        self.trajectories = simulation.simulate_random_ensemble(
            self.kernel, self.n_variables, self.n_time_steps, self.n_walkers,
            averages_only=averages_only, maskfunction=maskfunction, 
            threads_per_block=threads_per_block)
