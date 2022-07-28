from __future__ import annotations
from cubewalkers import simulation, parser

# for generating model names when the user doesn't want to specify them
import string
import random

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Experiment import Experiment


class Model():
    """Stores a Boolean network and experimental conditions, as well as generates and stores
    the results of simulations on that network.
    """
    _automatic_model_name_length = 16
    _automatic_model_names_taken = set()

    def __init__(self,
                 rules: str,
                 model_name: str | None = None,
                 experiment: Experiment | None = None,
                 comment_char: str = '#',
                 n_time_steps: int = 1,
                 n_walkers: int = 1) -> None:
        """Initializes Boolean network by generating a simulation kernel via 
        parser.bnet2rawkernel.

        Parameters
        ----------
        rules : str
            Rules to input. If skip_clean is True (not default), then these are assumed to 
            have been cleaned.
        kernel_name : str
            A name for the kernel
        experiment : Experiment | None, optional
            A string specifying experimental conditions, by default None, in which case no 
            experimental conditions are incorporated into the rules.
        comment_char : str, optional
            In rules, empty lines and lines beginning with this character are ignored, by 
            default '#'.
        n_time_steps: int, optional
            Number of timesteps to simulate, by default 1
        n_walkers: int, optional
            Number of ensemble walkers to simulate, by default 1
        """
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

    def simulate_random_ensemble(self, n_time_steps: int | None = None,
                                 n_walkers: int | None = None,
                                 averages_only: bool = False,
                                 maskfunction: callable | None = None,
                                 threads_per_block: tuple[int, int] = (32, 32)) -> None:
        """Simulates a random ensemble of walkers on the internally stored Boolean network.
        Results are stored in the trajectories attribute.

        Parameters
        ----------
        n_time_steps : int | None, optional
            If provided, the number of timesteps to simulate. Otherwise, uses internally
            stored value. By default None.
        n_walkers : int | None, optional
            If provided, the number of walkers to simulate. Otherwise, uses internally
            stored value. By default None.
        averages_only : bool, optional
            If True, stores only average node values at each timestep. 
            Otherwise, stores node values for each walker. By default False.
        maskfunction : callable, optional
            Function that returns a mask for selecting which node values to update. 
            By default, uses the synchronous update scheme. See update_schemes for examples.
        threads_per_block : tuple[int, int], optional
            How many threads should be in each block for each dimension of the N x W array, 
            by default (32, 32). See CUDA documentation for details.
        """
        if n_time_steps is not None:
            self.n_time_steps = n_time_steps
        if n_walkers is not None:
            self.n_walkers = n_walkers
        
        self.averages_only = averages_only
        
        self.trajectories = simulation.simulate_random_ensemble(
            self.kernel, self.n_variables, self.n_time_steps, self.n_walkers,
            averages_only=averages_only, maskfunction=maskfunction, 
            threads_per_block=threads_per_block)
