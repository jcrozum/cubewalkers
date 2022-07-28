from __future__ import annotations
from cubewalkers import simulation, parser, initial_conditions

# for generating model names when the user doesn't want to specify them
import string
import random

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Experiment import Experiment
    import cupy as cp


class Model():
    """Stores a Boolean network and experimental conditions, as well as generates and stores
    the results of simulations on that network.
    """
    _automatic_model_name_length = 16
    _automatic_model_names_taken = set()

    def __init__(self,
                 rules: str,
                 initial_biases: str = "",
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
        initial_biases : str
            Each line should be of the form
            NodeName,bias
            where NodeName is the name of the node, and bias is the probability that the node
            will be initialized to 1 (instead of 0). Nodes whose names are not listed are given
            a bias of 0.5 by default.
        model_name : str
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
        self.comment_char = comment_char
        self.initial_biases = initial_biases
        self.initialize_walkers()

    def initialize_walkers(self) -> None:
        """Generates initial conditions from internally stored data. See the 
        initial_conditions module for details.
        """
        self.initial_states = initial_conditions.initial_walker_states(
            self.initial_biases,
            self.vardict,
            self.n_walkers,
            comment_char=self.comment_char)

    def simulate_ensemble(self,
                          averages_only: bool = False,
                          maskfunction: callable | None = None,
                          threads_per_block: tuple[int, int] = (32, 32)) -> None:
        """Simulates a random ensemble of walkers on the internally stored Boolean network.
        Results are stored in the trajectories attribute.

        Parameters
        ----------
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

        if self.n_walkers != self.initial_states.shape[1]:
            self.initialize_walkers()

        self.averages_only = averages_only

        self.trajectories = simulation.simulate_ensemble(
            self.kernel, self.n_variables, self.n_time_steps, self.n_walkers,
            initial_states=self.initial_states,
            averages_only=averages_only,
            maskfunction=maskfunction,
            threads_per_block=threads_per_block)
